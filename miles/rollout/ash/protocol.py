from typing import Any, Literal

from pydantic import Field, model_validator

from miles.utils.pydantic_utils import FrozenStrictBaseModel

ASH_ROLLOUT_PROTOCOL_VERSION = "ash-rollout-v1"

AshJobStatus = Literal["queued", "running", "completed", "early_stopped", "failed", "cancelled"]
AshTrajectoryStatus = Literal["completed", "truncated", "failed", "aborted"]


class AshSampleSlot(FrozenStrictBaseModel):
    sample_slot_id: str = Field(min_length=1)
    sample_index: int = Field(ge=0)


class AshRolloutBudget(FrozenStrictBaseModel):
    max_model_calls: int = Field(gt=0)
    max_tool_calls: int = Field(ge=0)
    max_wall_time_seconds: float = Field(gt=0)


class AshRolloutRequest(FrozenStrictBaseModel):
    protocol_version: Literal[ASH_ROLLOUT_PROTOCOL_VERSION] = ASH_ROLLOUT_PROTOCOL_VERSION
    rollout_job_id: str = Field(min_length=1)
    rollout_id: int = Field(ge=0)
    prompt_group_id: str = Field(min_length=1)
    sample_slots: list[AshSampleSlot] = Field(min_length=1)
    max_samples: int = Field(gt=0)
    prompt: str | list[dict[str, Any]]
    prompt_token_ids: list[int] = Field(min_length=1)
    model_endpoint: str = Field(min_length=1)
    session_server_endpoint: str | None = Field(default=None, min_length=1)
    model: str | None = Field(default=None, min_length=1)
    expected_weight_version: str | None = Field(default=None, min_length=1)
    return_rollout_logprobs: bool = False
    sampling_params: dict[str, Any] = Field(default_factory=dict)
    budgets: AshRolloutBudget
    minimum_returned_samples: int = Field(default=1, ge=1)

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
    response_id: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    input_token_ids: list[int]
    output_token_ids: list[int] = Field(min_length=1)
    output_token_log_probs: list[float] | None = None
    weight_version: str = Field(min_length=1)
    finish_reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_lengths(self) -> "AshGeneratedSpan":
        if self.end <= self.start:
            raise ValueError("generated span end must be greater than start")
        span_length = self.end - self.start
        if len(self.output_token_ids) != span_length:
            raise ValueError("output_token_ids length must match the generated span")
        if self.output_token_log_probs is not None and len(self.output_token_log_probs) != span_length:
            raise ValueError("output_token_log_probs length must match the generated span")
        return self


class AshTrajectory(FrozenStrictBaseModel):
    sample_slot_id: str = Field(min_length=1)
    branch_id: str = Field(min_length=1)
    parent_branch_id: str | None = None
    branch_point_token_count: int | None = Field(default=None, ge=0)
    messages: list[dict[str, Any]] = Field(min_length=1)
    token_ids: list[int] = Field(min_length=1)
    prompt_length: int = Field(ge=1)
    generated_spans: list[AshGeneratedSpan] = Field(min_length=1)
    response_text: str
    reward: float | dict[str, Any] | None = None
    status: AshTrajectoryStatus
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_token_bounds(self) -> "AshTrajectory":
        if self.prompt_length >= len(self.token_ids):
            raise ValueError("prompt_length must leave at least one response token")
        return self


class AshRolloutSubmission(FrozenStrictBaseModel):
    protocol_version: Literal[ASH_ROLLOUT_PROTOCOL_VERSION] = ASH_ROLLOUT_PROTOCOL_VERSION
    rollout_job_id: str = Field(min_length=1)
    status: Literal["queued", "running"]


class AshRolloutDeletion(FrozenStrictBaseModel):
    protocol_version: Literal[ASH_ROLLOUT_PROTOCOL_VERSION] = ASH_ROLLOUT_PROTOCOL_VERSION
    rollout_job_id: str = Field(min_length=1)
    status: Literal["completed", "early_stopped", "failed", "cancelled"]


class AshRolloutResult(FrozenStrictBaseModel):
    protocol_version: Literal[ASH_ROLLOUT_PROTOCOL_VERSION] = ASH_ROLLOUT_PROTOCOL_VERSION
    rollout_job_id: str = Field(min_length=1)
    prompt_group_id: str = Field(min_length=1)
    status: AshJobStatus
    max_samples: int = Field(gt=0)
    actual_samples: int = Field(ge=0)
    stop_reason: str | None = None
    search_branches: int = Field(default=0, ge=0)
    consumed_budget: dict[str, float | int] = Field(default_factory=dict)
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
        return self
