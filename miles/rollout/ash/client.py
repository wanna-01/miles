import asyncio
from collections.abc import Mapping
from typing import Any

import httpx

from miles.rollout.ash.protocol import (
    AshRolloutDeletion,
    AshRolloutRequest,
    AshRolloutResult,
    AshRolloutSubmission,
)

_TERMINAL_JOB_STATUSES = {"completed", "early_stopped", "failed", "cancelled"}


class AshRolloutClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float | None = 30.0,
        headers: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout),
            headers=headers,
        )

    async def submit(self, request: AshRolloutRequest) -> AshRolloutSubmission:
        response = await self._client.post("/rollout-groups", json=request.model_dump(mode="json"))
        response.raise_for_status()
        return AshRolloutSubmission.model_validate(response.json())

    async def get_result(self, rollout_job_id: str) -> AshRolloutResult:
        response = await self._client.get(f"/rollout-groups/{rollout_job_id}")
        response.raise_for_status()
        return AshRolloutResult.model_validate(response.json())

    async def wait_for_result(
        self,
        rollout_job_id: str,
        *,
        poll_interval_seconds: float = 0.5,
    ) -> AshRolloutResult:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be greater than zero")

        while True:
            result = await self.get_result(rollout_job_id)
            if result.rollout_job_id != rollout_job_id:
                raise ValueError(
                    f"Ash returned rollout_job_id={result.rollout_job_id!r} " f"while polling {rollout_job_id!r}"
                )
            if result.status in _TERMINAL_JOB_STATUSES:
                return result
            await asyncio.sleep(poll_interval_seconds)

    async def delete(self, rollout_job_id: str) -> AshRolloutDeletion:
        """Cancel unfinished work and release Ash's job record."""
        response = await self._client.delete(f"/rollout-groups/{rollout_job_id}")
        response.raise_for_status()
        return AshRolloutDeletion.model_validate(response.json())

    async def cancel(self, rollout_job_id: str) -> AshRolloutDeletion:
        """Compatibility alias for the original client API."""
        return await self.delete(rollout_job_id)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "AshRolloutClient":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()
