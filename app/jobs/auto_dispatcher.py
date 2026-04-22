"""Auto-fallback dispatcher: tries HPC first, falls back to local on failure.

Uses HPCDispatcher as the primary backend. If HPC is unreachable (SSH fails,
sbatch errors), resubmits the job to InProcessQueue for local processing.
Periodically checks if HPC is back online.
"""

import asyncio
import logging
from typing import Optional

from app.jobs.dispatcher import JobDispatcher
from app.jobs.models import JobRecord, JobStatus

logger = logging.getLogger(__name__)

# How often to check if HPC is back online after a failure (seconds)
HPC_HEALTH_CHECK_INTERVAL = 120


class AutoDispatcher(JobDispatcher):
    """Wraps a primary (HPC) and fallback (local) dispatcher.

    Submits to HPC by default. If HPC submission fails, transparently
    resubmits to the local fallback. Periodically probes HPC health.
    """

    def __init__(self, primary: JobDispatcher, fallback: JobDispatcher):
        self._primary = primary
        self._fallback = fallback
        self._hpc_healthy = True
        self._health_task: Optional[asyncio.Task] = None

    async def submit(self, job: JobRecord) -> str:
        """Submit a job. Tries HPC first, falls back to local if HPC is down."""
        if self._hpc_healthy:
            try:
                return await self._primary.submit(job)
            except Exception as e:
                logger.warning(
                    "HPC submission failed for job %s, falling back to local: %s",
                    job.id, str(e),
                )
                self._hpc_healthy = False
                # Reset job state for local processing
                job.status = JobStatus.PENDING
                job.dispatch_backend = "local"
                job.slurm_job_id = None
                job.error = None

        return await self._fallback.submit(job)

    async def get_status(self, job_id: str) -> Optional[JobRecord]:
        """Check both dispatchers for the job."""
        result = await self._primary.get_status(job_id)
        if result is not None:
            return result
        return await self._fallback.get_status(job_id)

    async def start(self) -> None:
        """Start both dispatchers and the health check loop."""
        await self._primary.start()
        await self._fallback.start()
        self._health_task = asyncio.create_task(self._health_check_loop())
        logger.info("AutoDispatcher started (HPC primary, local fallback)")

    async def stop(self) -> None:
        """Stop both dispatchers and the health check."""
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        await self._primary.stop()
        await self._fallback.stop()
        logger.info("AutoDispatcher stopped")

    async def _health_check_loop(self) -> None:
        """Periodically check if HPC is reachable when it's marked down."""
        while True:
            try:
                await asyncio.sleep(HPC_HEALTH_CHECK_INTERVAL)

                if self._hpc_healthy:
                    continue  # Already healthy, skip check

                # Try to connect
                try:
                    from app.jobs.ssh_client import HPCSSHClient
                    probe = HPCSSHClient()
                    await probe.connect()
                    _, _, exit_code = await probe.run_command("echo hpc_ok", timeout=10)
                    await probe.close()

                    if exit_code == 0:
                        self._hpc_healthy = True
                        logger.info("HPC is back online — switching to HPC dispatch")
                except Exception:
                    pass  # Still down

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check error: %s", str(e))
