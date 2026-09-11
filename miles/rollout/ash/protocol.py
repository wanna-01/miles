import math
import re
from typing import Annotated, Any, Literal

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StringConstraints, model_validator

from miles.utils.pydantic_utils import FrozenStrictBaseModel

ASH_ROLLOUT_PROTOCOL_VERSION = "ash-rollout-v2"

NonEmptyStr = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
StrictNumber = StrictFloat | StrictInt

AshJobStatus = Literal["queued", "running", "completed", "early_stopped", "failed", "cancelled"]
AshTrajectoryStatus = Literal["completed", "truncated", "failed", "aborted"]


class AshSampleSlot(FrozenStrictBaseModel):
    sample_slot_id: NonEmptyStr
    sample_index: StrictInt = Field(ge=0)


class AshRolloutBudget(FrozenStrictBaseModel):
    max_model_calls: StrictInt = Field(gt=0)
    max_tool_calls: StrictInt = Field(ge=0)
    max_wall_time_seconds: StrictFloat = Field(gt=0)

    @model_validator(mode="after")
    def validate_wall_time(self) -> "AshRolloutBudget":
        if not math.isfinite(self.max_wall_time_seconds):
            raise ValueError("max_wall_time_seconds must be finite")
        return self


class AshEnvironmentRef(FrozenStrictBaseModel):
    kind: Literal["image", "template", "snapshot"]
    id: NonEmptyStr
    revision: NonEmptyStr
    resource_profile: NonEmptyStr

    @model_validator(mode="after")
    def validate_image_revision(self) -> "AshEnvironmentRef":
        if self.kind == "image" and not re.fullmatch(
            r"sha256:[0-9a-fA-F]{64}", self.revision
        ):
            raise ValueError(
                "revision must be a sha256 digest when kind is image"
            )
        return self


class AshEnvironmentList(FrozenStrictBaseModel):
    protocol_version: Literal[ASH_ROLLOUT_PROTOCOL_VERSION] = ASH_ROLLOUT_PROTOCOL_VERSION
    environments: list[AshEnvironmentRef] = Field(default_factory=list)


class AshRolloutRequest(FrozenStrictBaseModel):
    protocol_version: Literal[ASH_ROLLOUT_PROTOCOL_VERSION] = ASH_ROLLOUT_PROTOCOL_VERSION
    rollout_job_id: NonEmptyStr
    rollout_id: StrictInt = Field(ge=0)
    prompt_group_id: NonEmptyStr
    task_id: NonEmptyStr
    environment_ref: AshEnvironmentRef
    sample_slots: list[AshSampleSlot] = Field(min_length=1)
    max_samples: StrictInt = Field(gt=0)
    prompt: str | list[dict[str, Any]]
    prompt_token_ids: list[StrictInt] = Field(min_length=1)
    model_endpoint: NonEmptyStr
    session_server_endpoint: NonEmptyStr | None = None
    model: NonEmptyStr | None = None
    expected_weight_version: NonEmptyStr | None = None
    return_rollout_logprobs: StrictBool = False
    sampling_params: dict[str, Any] = Field(default_factory=dict)
    budgets: AshRolloutBudget
    minimum_returned_samples: StrictInt = Field(default=1, ge=1)

    @model_validator(mode="after")
    def validate_slots(self) -> "AshRolloutRequest":
        slot_ids = [slot.sample_slot_id for slot in self.sample_slots]
        sample_indices = [slot.sample_index for slot in self.sample_slots]
        if len(slot_ids) != len(set(slot_ids)):
            raise ValueError("sample_slot_id values must be unique")
        if len(sample_indices) != len(set(sample_indices)):
            raise ValueError("sample_index values must be unique")
        if self.max_samples > len(self.sample_slots):
            raise ValueError("max_samples cannot exceed the number of sample slots")
        if self.minimum_returned_samples > self.max_samples:
            raise ValueError("minimum_returned_samples cannot exceed max_samples")
        return self


class AshGeneratedSpan(FrozenStrictBaseModel):
    response_id: NonEmptyStr
    start: StrictInt = Field(ge=0)
    end: StrictInt = Field(gt=0)
    input_token_ids: list[StrictInt]
    output_token_ids: list[StrictInt] = Field(min_length=1)
    output_token_log_probs: list[StrictNumber] | None = None
    weight_version: NonEmptyStr
    finish_reason: NonEmptyStr

    @model_validator(mode="after")
    def validate_lengths(self) -> "AshGeneratedSpan":
        if self.end <= self.start:
            raise ValueError("generated span end must be greater than start")
        span_length = self.end - self.start
        if len(self.output_token_ids) != span_length:
            raise ValueError("output_token_ids length must match the generated span")
        if self.output_token_log_probs is not None and len(self.output_token_log_probs) != span_length:
            raise ValueError("output_token_log_probs length must match the generated span")
        if self.output_token_log_probs is not None and any(
            not math.isfinite(value) for value in self.output_token_log_probs
        ):
            raise ValueError("output_token_log_probs must contain finite numbers")
        return self


class AshTrajectory(FrozenStrictBaseModel):
    sample_slot_id: NonEmptyStr
    branch_id: NonEmptyStr
    parent_branch_id: NonEmptyStr | None = None
    branch_point_token_count: StrictInt | None = Field(default=None, ge=0)
    messages: list[dict[str, Any]] = Field(min_length=1)
    token_ids: list[StrictInt] = Field(min_length=1)
    prompt_length: StrictInt = Field(ge=1)
    generated_spans: list[AshGeneratedSpan] = Field(min_length=1)
    response_text: str
    reward: StrictNumber | dict[str, Any] | None = None
    status: AshTrajectoryStatus
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_token_bounds(self) -> "AshTrajectory":
        if self.prompt_length >= len(self.token_ids):
            raise ValueError("prompt_length must leave at least one response token")
        if isinstance(self.reward, (int, float)) and not math.isfinite(self.reward):
            raise ValueError("numeric reward must be finite")
        return self


class AshRolloutSubmission(FrozenStrictBaseModel):
    protocol_version: Literal[ASH_ROLLOUT_PROTOCOL_VERSION] = ASH_ROLLOUT_PROTOCOL_VERSION
    rollout_job_id: NonEmptyStr
    # Idempotent POST retries report the job's current state. The original
    # acknowledgement is normally queued/running, but the job may already be
    # terminal when a client retries after losing the first response.
    status: AshJobStatus


class AshRolloutDeletion(FrozenStrictBaseModel):
    protocol_version: Literal[ASH_ROLLOUT_PROTOCOL_VERSION] = ASH_ROLLOUT_PROTOCOL_VERSION
    rollout_job_id: NonEmptyStr
    status: Literal["completed", "early_stopped", "failed", "cancelled"]


class AshRolloutResult(FrozenStrictBaseModel):
    protocol_version: Literal[ASH_ROLLOUT_PROTOCOL_VERSION] = ASH_ROLLOUT_PROTOCOL_VERSION
    rollout_job_id: NonEmptyStr
    prompt_group_id: NonEmptyStr
    status: AshJobStatus
    max_samples: StrictInt = Field(gt=0)
    actual_samples: StrictInt = Field(ge=0)
    stop_reason: str | None = None
    search_branches: StrictInt = Field(default=0, ge=0)
    consumed_budget: dict[str, StrictNumber] = Field(default_factory=dict)
    trajectories: list[AshTrajectory] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_trajectories(self) -> "AshRolloutResult":
        if self.actual_samples != len(self.trajectories):
            raise ValueError("actual_samples must match the number of trajectories")
        if self.actual_samples > self.max_samples:
            raise ValueError("actual_samples cannot exceed max_samples")
        slot_ids = [trajectory.sample_slot_id for trajectory in self.trajectories]
        branch_ids = [trajectory.branch_id for trajectory in self.trajectories]
        if len(slot_ids) != len(set(slot_ids)):
            raise ValueError("returned sample_slot_id values must be unique")
        if len(branch_ids) != len(set(branch_ids)):
            raise ValueError("returned branch_id values must be unique")
        if self.status in {"queued", "running"} and self.trajectories:
            raise ValueError("non-terminal rollout results cannot contain trajectories")
        for name, value in self.consumed_budget.items():
            if not name.strip():
                raise ValueError("consumed_budget keys must be non-empty strings")
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"consumed_budget[{name!r}] must be a finite non-negative number"
                )
        return self
