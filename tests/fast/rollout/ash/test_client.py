import asyncio

import httpx

from miles.rollout.ash.client import AshRolloutClient
from miles.rollout.ash.protocol import AshRolloutBudget, AshRolloutRequest, AshSampleSlot


def _request():
    return AshRolloutRequest(
        rollout_job_id="job-1",
        rollout_id=4,
        prompt_group_id="group-3",
        sample_slots=[AshSampleSlot(sample_slot_id="slot-11", sample_index=11)],
        max_samples=1,
        prompt=[{"role": "user", "content": "fix it"}],
        prompt_token_ids=[10],
        model_endpoint="http://miles/v1",
        expected_weight_version="7",
        budgets=AshRolloutBudget(max_model_calls=10, max_tool_calls=10, max_wall_time_seconds=60),
    )


def test_client_submit_get_and_cancel():
    methods = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append((request.method, request.url.path))
        if request.method == "POST":
            return httpx.Response(
                202,
                json={
                    "protocol_version": "ash-rollout-v1",
                    "rollout_job_id": "job-1",
                    "status": "queued",
                },
            )
        status = "cancelled" if request.method == "DELETE" else "running"
        return httpx.Response(
            200,
            json={
                "protocol_version": "ash-rollout-v1",
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
            cancelled = await client.cancel("job-1")
            return submission, running, cancelled
        finally:
            await async_client.aclose()

    submission, running, cancelled = asyncio.run(exercise_client())

    assert submission.status == "queued"
    assert running.status == "running"
    assert cancelled.status == "cancelled"
    assert methods == [
        ("POST", "/rollout-groups"),
        ("GET", "/rollout-groups/job-1"),
        ("DELETE", "/rollout-groups/job-1"),
    ]


def test_client_waits_until_terminal_result():
    polls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal polls
        polls += 1
        status = "running" if polls == 1 else "completed"
        return httpx.Response(
            200,
            json={
                "protocol_version": "ash-rollout-v1",
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
