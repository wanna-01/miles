from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from miles.rollout.ash.protocol import AshGeneratedSpan, AshRolloutResult, AshTrajectory
from miles.utils.types import Sample

_TERMINAL_JOB_STATUSES = {"completed", "early_stopped", "failed", "cancelled"}
_SAMPLE_STATUS = {
    "completed": Sample.Status.COMPLETED,
    "truncated": Sample.Status.TRUNCATED,
    "failed": Sample.Status.FAILED,
    "aborted": Sample.Status.ABORTED,
}


def import_ash_rollout_result(
    result: AshRolloutResult,
    slot_samples: Mapping[str, Sample],
) -> list[Sample]:
    """Validate an Ash result and convert its returned leaves into Miles samples."""
    if result.status not in _TERMINAL_JOB_STATUSES:
        raise ValueError(f"cannot import non-terminal Ash rollout with status={result.status!r}")
    if result.max_samples > len(slot_samples):
        raise ValueError("Ash max_samples exceeds the sample slots allocated by Miles")

    imported = []
    for trajectory in result.trajectories:
        if trajectory.sample_slot_id not in slot_samples:
            raise ValueError(f"unknown sample_slot_id returned by Ash: {trajectory.sample_slot_id!r}")
        imported.append(
            _import_trajectory(
                result=result,
                trajectory=trajectory,
                input_sample=slot_samples[trajectory.sample_slot_id],
            )
        )
    return imported


def _import_trajectory(
    *,
    result: AshRolloutResult,
    trajectory: AshTrajectory,
    input_sample: Sample,
) -> Sample:
    _validate_messages(trajectory.messages)
    _validate_branch_point(trajectory)

    loss_mask, rollout_log_probs, weight_versions = _build_token_metadata(trajectory)
    prompt_tokens = trajectory.token_ids[: trajectory.prompt_length]
    if (
        trajectory.prompt_token_alignment == "request_exact"
        and input_sample.tokens
        and input_sample.tokens != prompt_tokens
    ):
        raise ValueError(
            f"Ash prompt tokens do not match Miles sample {trajectory.sample_slot_id!r}: "
            f"expected {input_sample.tokens}, got {prompt_tokens}"
        )

    sample = deepcopy(input_sample)
    sample.tokens = list(trajectory.token_ids)
    sample.response = trajectory.response_text
    sample.response_length = len(trajectory.token_ids) - trajectory.prompt_length
    sample.loss_mask = loss_mask
    sample.rollout_log_probs = rollout_log_probs
    sample.weight_versions = weight_versions
    sample.reward = trajectory.reward
    sample.status = _SAMPLE_STATUS[trajectory.status]

    lineage = {
        "protocol_version": result.protocol_version,
        "rollout_job_id": result.rollout_job_id,
        "prompt_group_id": result.prompt_group_id,
        "sample_slot_id": trajectory.sample_slot_id,
        "branch_id": trajectory.branch_id,
        "parent_branch_id": trajectory.parent_branch_id,
        "branch_point_token_count": trajectory.branch_point_token_count,
        "prompt_token_alignment": trajectory.prompt_token_alignment,
        "request_prompt_token_ids": list(input_sample.tokens),
        "messages": deepcopy(trajectory.messages),
        "trajectory_metadata": deepcopy(trajectory.metadata),
        "stop_reason": result.stop_reason,
        "search_branches": result.search_branches,
        "consumed_budget": dict(result.consumed_budget),
    }
    sample.metadata = {**sample.metadata, "ash_rollout": lineage}
    sample.train_metadata = {**(sample.train_metadata or {}), "ash_rollout": _training_lineage(lineage)}
    sample.validate()
    return sample


def _build_token_metadata(trajectory: AshTrajectory) -> tuple[list[int], list[float] | None, list[str]]:
    response_length = len(trajectory.token_ids) - trajectory.prompt_length
    loss_mask = [0] * response_length
    span_log_prob_presence = [span.output_token_log_probs is not None for span in trajectory.generated_spans]
    if any(span_log_prob_presence) and not all(span_log_prob_presence):
        raise ValueError("generated spans must either all include output_token_log_probs or all omit them")
    rollout_log_probs = [0.0] * response_length if all(span_log_prob_presence) else None
    weight_versions = []
    previous_end = trajectory.prompt_length

    for span in trajectory.generated_spans:
        _validate_generated_span(trajectory, span, previous_end=previous_end)
        response_start = span.start - trajectory.prompt_length
        response_end = span.end - trajectory.prompt_length
        loss_mask[response_start:response_end] = [1] * len(span.output_token_ids)
        if rollout_log_probs is not None:
            assert span.output_token_log_probs is not None
            rollout_log_probs[response_start:response_end] = span.output_token_log_probs
        weight_versions.append(span.weight_version)
        previous_end = span.end

    return loss_mask, rollout_log_probs, weight_versions


def _validate_generated_span(
    trajectory: AshTrajectory,
    span: AshGeneratedSpan,
    *,
    previous_end: int,
) -> None:
    if span.start < trajectory.prompt_length:
        raise ValueError(f"generated span {span.response_id!r} overlaps the prompt")
    if span.start < previous_end:
        raise ValueError(f"generated span {span.response_id!r} overlaps or is out of order")
    if span.end > len(trajectory.token_ids):
        raise ValueError(f"generated span {span.response_id!r} exceeds the trajectory token sequence")
    if span.input_token_ids != trajectory.token_ids[: span.start]:
        raise ValueError(f"input_token_ids mismatch for response {span.response_id!r}")
    if span.output_token_ids != trajectory.token_ids[span.start : span.end]:
        raise ValueError(f"output_token_ids mismatch for response {span.response_id!r}")


def _validate_branch_point(trajectory: AshTrajectory) -> None:
    branch_point = trajectory.branch_point_token_count
    if branch_point is not None and branch_point > len(trajectory.token_ids):
        raise ValueError("branch_point_token_count exceeds the trajectory token sequence")
    if trajectory.parent_branch_id is None and branch_point is not None:
        raise ValueError("a root trajectory cannot declare branch_point_token_count")
    if trajectory.parent_branch_id is not None and branch_point is None:
        raise ValueError("a child trajectory must declare branch_point_token_count")


def _validate_messages(messages: list[dict[str, Any]]) -> None:
    known_tool_calls: set[str] = set()
    consumed_tool_calls: set[str] = set()

    for position, message in enumerate(messages):
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role at position {position}: {role!r}")

        if role == "assistant":
            for tool_call in message.get("tool_calls") or []:
                tool_call_id = tool_call.get("id")
                if not tool_call_id:
                    raise ValueError(f"assistant tool call at position {position} has no id")
                if tool_call_id in known_tool_calls:
                    raise ValueError(f"duplicate assistant tool_call_id: {tool_call_id!r}")
                known_tool_calls.add(tool_call_id)

        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if not tool_call_id:
                raise ValueError(f"tool result at position {position} has no tool_call_id")
            if tool_call_id not in known_tool_calls:
                raise ValueError(f"tool result references an unknown tool_call_id: {tool_call_id!r}")
            if tool_call_id in consumed_tool_calls:
                raise ValueError(f"duplicate tool result for tool_call_id: {tool_call_id!r}")
            consumed_tool_calls.add(tool_call_id)

    missing_results = known_tool_calls - consumed_tool_calls
    if missing_results:
        raise ValueError(f"assistant tool calls have no matching tool results: {sorted(missing_results)}")


def _training_lineage(lineage: dict[str, Any]) -> dict[str, Any]:
    return {
        "rollout_job_id": lineage["rollout_job_id"],
        "prompt_group_id": lineage["prompt_group_id"],
        "sample_slot_id": lineage["sample_slot_id"],
        "branch_id": lineage["branch_id"],
        "parent_branch_id": lineage["parent_branch_id"],
        "branch_point_token_count": lineage["branch_point_token_count"],
    }
