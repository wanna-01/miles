from argparse import Namespace

import pytest
from pydantic import ValidationError

from miles.rollout.ash.importer import import_ash_rollout_result
from miles.rollout.ash.protocol import AshRolloutResult
from miles.utils.types import Sample


def _result_payload():
    return {
        "protocol_version": "ash-rollout-v2",
        "rollout_job_id": "job-1",
        "prompt_group_id": "group-3",
        "status": "early_stopped",
        "max_samples": 4,
        "actual_samples": 1,
        "stop_reason": "positive_threshold_reached",
        "search_branches": 3,
        "consumed_budget": {"model_calls": 2, "tool_calls": 1},
        "trajectories": [
            {
                "sample_slot_id": "slot-11",
                "branch_id": "branch-a",
                "parent_branch_id": None,
                "branch_point_token_count": None,
                "messages": [
                    {"role": "user", "content": "fix it"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "call-1", "type": "function", "function": {}}],
                    },
                    {"role": "tool", "tool_call_id": "call-1", "content": "done"},
                    {"role": "assistant", "content": "fixed"},
                ],
                "token_ids": [10, 11, 20, 21, 30, 31, 40],
                "prompt_length": 2,
                "generated_spans": [
                    {
                        "response_id": "response-1",
                        "start": 2,
                        "end": 4,
                        "input_token_ids": [10, 11],
                        "output_token_ids": [20, 21],
                        "output_token_log_probs": [-0.1, -0.2],
                        "weight_version": "7",
                        "finish_reason": "tool_calls",
                    },
                    {
                        "response_id": "response-2",
                        "start": 6,
                        "end": 7,
                        "input_token_ids": [10, 11, 20, 21, 30, 31],
                        "output_token_ids": [40],
                        "output_token_log_probs": [-0.3],
                        "weight_version": "7",
                        "finish_reason": "stop",
                    },
                ],
                "response_text": "tool call, tool result, fixed",
                "reward": 1.0,
                "status": "completed",
                "metadata": {"environment_checkpoint_ids": ["checkpoint-1"]},
            }
        ],
    }


def _slot_samples(*, include_returned_slot=True):
    samples = {
        "slot-12": Sample(index=12),
        "slot-13": Sample(index=13),
        "slot-14": Sample(index=14),
        "slot-15": Sample(index=15),
    }
    if include_returned_slot:
        samples["slot-11"] = Sample(prompt="fix it", tokens=[10, 11], group_index=3, index=11)
        samples.pop("slot-15")
    return samples


def test_imports_partial_group_and_masks_tool_observation():
    result = AshRolloutResult.model_validate(_result_payload())

    (sample,) = import_ash_rollout_result(result, _slot_samples())

    assert sample.tokens == [10, 11, 20, 21, 30, 31, 40]
    assert sample.response_length == 5
    assert sample.loss_mask == [1, 1, 0, 0, 1]
    assert sample.rollout_log_probs == [-0.1, -0.2, 0.0, 0.0, -0.3]
    assert sample.weight_versions == ["7", "7"]
    assert sample.group_index == 3
    assert sample.index == 11
    assert sample.rollout_id is None
    assert sample.reward == 1.0
    assert sample.metadata["ash_rollout"]["search_branches"] == 3
    assert sample.train_metadata["ash_rollout"]["branch_id"] == "branch-a"


def test_imports_trajectory_without_rollout_log_probs():
    payload = _result_payload()
    for span in payload["trajectories"][0]["generated_spans"]:
        span.pop("output_token_log_probs")
    result = AshRolloutResult.model_validate(payload)

    (sample,) = import_ash_rollout_result(result, _slot_samples())

    assert sample.rollout_log_probs is None
    assert sample.loss_mask == [1, 1, 0, 0, 1]
    assert sample.weight_versions == ["7", "7"]


def test_imports_harness_rendered_prompt_and_preserves_requested_prompt_tokens():
    payload = _result_payload()
    trajectory = payload["trajectories"][0]
    trajectory["prompt_token_alignment"] = "harness_rendered"
    trajectory["token_ids"][:2] = [100, 101]
    for span in trajectory["generated_spans"]:
        span["input_token_ids"][:2] = [100, 101]
    result = AshRolloutResult.model_validate(payload)

    (sample,) = import_ash_rollout_result(result, _slot_samples())

    assert sample.tokens[:2] == [100, 101]
    assert sample.metadata["ash_rollout"]["prompt_token_alignment"] == "harness_rendered"
    assert sample.metadata["ash_rollout"]["request_prompt_token_ids"] == [10, 11]


def test_request_exact_prompt_still_rejects_a_token_mismatch():
    payload = _result_payload()
    trajectory = payload["trajectories"][0]
    trajectory["token_ids"][:2] = [100, 101]
    for span in trajectory["generated_spans"]:
        span["input_token_ids"][:2] = [100, 101]
    result = AshRolloutResult.model_validate(payload)

    with pytest.raises(ValueError, match="prompt tokens do not match"):
        import_ash_rollout_result(result, _slot_samples())


def test_rejects_mixed_rollout_log_prob_presence():
    payload = _result_payload()
    payload["trajectories"][0]["generated_spans"][1].pop("output_token_log_probs")
    result = AshRolloutResult.model_validate(payload)

    with pytest.raises(ValueError, match="either all include output_token_log_probs or all omit"):
        import_ash_rollout_result(result, _slot_samples())


@pytest.mark.parametrize("include_rollout_log_probs", [False, True])
def test_imported_sample_reaches_train_data_conversion(include_rollout_log_probs):
    pytest.importorskip("ray")
    from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data

    payload = _result_payload()
    if not include_rollout_log_probs:
        for span in payload["trajectories"][0]["generated_spans"]:
            span.pop("output_token_log_probs")
    result = AshRolloutResult.model_validate(payload)
    samples = import_ash_rollout_result(result, _slot_samples())

    train_data = convert_samples_to_train_data(
        Namespace(
            advantage_estimator="grpo",
            rewards_normalization=False,
            reward_key=None,
            use_dynamic_global_batch_size=False,
        ),
        samples,
        metadata={},
        custom_convert_samples_to_train_data_func=None,
        custom_reward_post_process_func=None,
    )

    assert ("rollout_log_probs" in train_data) is include_rollout_log_probs
    assert train_data["tokens"] == [[10, 11, 20, 21, 30, 31, 40]]
    assert train_data["loss_masks"] == [[1, 1, 0, 0, 1]]
    assert train_data["weight_versions"] == [["7", "7"]]


def test_rejects_unknown_sample_slot():
    result = AshRolloutResult.model_validate(_result_payload())

    with pytest.raises(ValueError, match="unknown sample_slot_id"):
        import_ash_rollout_result(result, _slot_samples(include_returned_slot=False))


def test_rejects_output_token_mismatch():
    payload = _result_payload()
    payload["trajectories"][0]["generated_spans"][1]["output_token_ids"] = [99]
    result = AshRolloutResult.model_validate(payload)

    with pytest.raises(ValueError, match="output_token_ids mismatch"):
        import_ash_rollout_result(result, _slot_samples())


def test_rejects_duplicate_returned_slot():
    payload = _result_payload()
    payload["trajectories"].append(payload["trajectories"][0].copy())
    payload["trajectories"][1]["branch_id"] = "branch-b"
    payload["actual_samples"] = 2

    with pytest.raises(ValidationError, match="returned sample_slot_id values must be unique"):
        AshRolloutResult.model_validate(payload)


def test_rejects_unmatched_tool_call():
    payload = _result_payload()
    payload["trajectories"][0]["messages"] = payload["trajectories"][0]["messages"][:2]
    result = AshRolloutResult.model_validate(payload)

    with pytest.raises(ValueError, match="no matching tool results"):
        import_ash_rollout_result(result, _slot_samples())
