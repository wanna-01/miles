import asyncio

import httpx
import pytest
from pydantic import ValidationError

from miles.rollout.ash.client import AshRolloutClient
from miles.rollout.ash.protocol import (
    AshEnvironmentRef,
    AshGeneratedSpan,
    AshRolloutBudget,
    AshRolloutRequest,
    AshRolloutResult,
    AshSampleSlot,
)


def _request():
    return AshRolloutRequest(
        rollout_job_id="job-1",
        rollout_id=4,
        prompt_group_id="group-3",
        task_id="swebench__repo-123",
        environment_ref=AshEnvironmentRef(
            kind="template",
            id="swebench-runtime",
            revision="sha256:test",
            resource_profile="standard",
        ),
        sample_slots=[AshSampleSlot(sample_slot_id="slot-11", sample_index=11)],
        max_samples=1,
        prompt=[{"role": "user", "content": "fix it"}],
        prompt_token_ids=[10],
        model_endpoint="http://miles/v1",
        expected_weight_version="7",
        budgets=AshRolloutBudget(max_model_calls=10, max_tool_calls=10, max_wall_time_seconds=60),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("session_server_endpoint", ""),
        ("model_endpoint", "   "),
        ("task_id", "\t"),
    ],
)
def test_request_rejects_empty_or_blank_strings(field, value):
    payload = _request().model_dump(mode="json")
    payload[field] = value
    with pytest.raises(ValidationError):
        AshRolloutRequest.model_validate(payload)


def test_request_rejects_unknown_fields():
    payload = _request().model_dump(mode="json")
    payload["future_request_field"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AshRolloutRequest.model_validate(payload)


def test_request_rejects_boolean_integer_fields():
    payload = _request().model_dump(mode="json")
    payload["sample_slots"][0]["sample_index"] = True
    with pytest.raises(ValidationError):
        AshRolloutRequest.model_validate(payload)


@pytest.mark.parametrize("value", [True, float("inf"), float("nan")])
def test_budget_rejects_non_finite_or_boolean_wall_time(value):
    with pytest.raises(ValidationError):
        AshRolloutBudget(
            max_model_calls=1,
            max_tool_calls=0,
            max_wall_time_seconds=value,
        )


@pytest.mark.parametrize("value", [True, -1, float("inf"), float("nan")])
def test_result_rejects_invalid_consumed_budget(value):
    with pytest.raises(ValidationError):
        AshRolloutResult(
            rollout_job_id="job-1",
            prompt_group_id="group-3",
            status="completed",
            max_samples=1,
            actual_samples=0,
            consumed_budget={"model_calls": value},
        )


@pytest.mark.parametrize("value", [True, float("inf"), float("nan")])
def test_generated_span_rejects_invalid_log_prob(value):
    with pytest.raises(ValidationError):
        AshGeneratedSpan(
            response_id="response-1",
            start=1,
            end=2,
            input_token_ids=[10],
            output_token_ids=[11],
            output_token_log_probs=[value],
            weight_version="1",
            finish_reason="stop",
        )


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_trajectory_rejects_non_finite_numeric_reward(value):
    from miles.rollout.ash.protocol import AshTrajectory

    with pytest.raises(ValidationError, match="numeric reward must be finite"):
        AshTrajectory(
            sample_slot_id="slot-1",
            branch_id="root",
            messages=[{"role": "user", "content": "hello"}],
            token_ids=[10, 11],
            prompt_length=1,
            generated_spans=[
                AshGeneratedSpan(
                    response_id="response-1",
                    start=1,
                    end=2,
                    input_token_ids=[10],
                    output_token_ids=[11],
                    weight_version="1",
                    finish_reason="stop",
                )
            ],
            response_text="world",
            reward=value,
            status="completed",
        )


def test_image_environment_requires_an_immutable_oci_digest():
    with pytest.raises(ValidationError, match="sha256 digest"):
        AshEnvironmentRef(
            kind="image",
            id="docker.io/example/task-environment",
            revision="latest",
            resource_profile="standard",
        )

    ref = AshEnvironmentRef(
        kind="image",
        id="docker.io/example/task-environment",
        revision="sha256:" + "a" * 64,
        resource_profile="standard",
    )

    assert ref.revision == "sha256:" + "a" * 64


def test_client_submit_get_and_delete():
    methods = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append((request.method, request.url.path))
        if request.method == "POST":
            return httpx.Response(
                202,
                json={
                    "protocol_version": "ash-rollout-v2",
                    "rollout_job_id": "job-1",
                    "status": "queued",
                },
            )
        status = "cancelled" if request.method == "DELETE" else "running"
        if request.method == "DELETE":
            return httpx.Response(
                200,
                json={
                    "protocol_version": "ash-rollout-v2",
                    "rollout_job_id": "job-1",
                    "status": status,
                },
            )
        return httpx.Response(
            200,
            json={
                "protocol_version": "ash-rollout-v2",
                "rollout_job_id": "job-1",
                "prompt_group_id": "group-3",
                "status": status,
                "max_samples": 1,
                "actual_samples": 0,
                "trajectories": [],
            },
        )

    async def exercise_client():
        async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://ash")
        client = AshRolloutClient("http://ash", client=async_client)
        try:
            submission = await client.submit(_request())
            running = await client.get_result("job-1")
            deleted = await client.delete("job-1")
            return submission, running, deleted
        finally:
            await async_client.aclose()

    submission, running, deleted = asyncio.run(exercise_client())

    assert submission.status == "queued"
    assert running.status == "running"
    assert deleted.status == "cancelled"
    assert methods == [
        ("POST", "/rollout-groups"),
        ("GET", "/rollout-groups/job-1"),
        ("DELETE", "/rollout-groups/job-1"),
    ]


def test_client_lists_static_environments_without_backend_handles():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/rollout-environments"
        return httpx.Response(
            200,
            json={
                "protocol_version": "ash-rollout-v2",
                "environments": [
                    {
                        "kind": "template",
                        "id": "swebench-runtime",
                        "revision": "sha256:immutable-revision",
                        "resource_profile": "standard",
                    }
                ],
            },
        )

    async def exercise_client():
        async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://ash")
        client = AshRolloutClient("http://ash", client=async_client)
        try:
            return await client.list_environments()
        finally:
            await async_client.aclose()

    result = asyncio.run(exercise_client())

    assert result.environments == [
        AshEnvironmentRef(
            kind="template",
            id="swebench-runtime",
            revision="sha256:immutable-revision",
            resource_profile="standard",
        )
    ]


def test_client_accepts_terminal_status_from_idempotent_submit_retry():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            202,
            json={
                "protocol_version": "ash-rollout-v2",
                "rollout_job_id": "job-1",
                "status": "completed",
            },
        )

    async def exercise_client():
        async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://ash")
        client = AshRolloutClient("http://ash", client=async_client)
        try:
            return await client.submit(_request())
        finally:
            await async_client.aclose()

    assert asyncio.run(exercise_client()).status == "completed"


def test_client_waits_until_terminal_result():
    polls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal polls
        polls += 1
        status = "running" if polls == 1 else "completed"
        return httpx.Response(
            200,
            json={
                "protocol_version": "ash-rollout-v2",
                "rollout_job_id": "job-1",
                "prompt_group_id": "group-3",
                "status": status,
                "max_samples": 1,
                "actual_samples": 0,
                "trajectories": [],
            },
        )

    async def exercise_client():
        async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://ash")
        client = AshRolloutClient("http://ash", client=async_client)
        try:
            return await client.wait_for_result("job-1", poll_interval_seconds=0.001)
        finally:
            await async_client.aclose()

    result = asyncio.run(exercise_client())

    assert result.status == "completed"
    assert polls == 2
