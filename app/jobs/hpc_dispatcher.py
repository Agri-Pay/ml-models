"""HPC job dispatcher — submits inference jobs to SLURM via SSH/SCP.

Implements the JobDispatcher ABC. Handles:
- SCP image upload to HPC scratch space
- SLURM script generation and submission (sbatch)
- Status polling via sacct
- Result download and local post-processing
- HPC scratch cleanup after completion
- Concurrency management (max 3 SLURM jobs per user)
"""

import asyncio
import json
import logging
import os
import tempfile
import time
import traceback
from datetime import datetime
from typing import Callable, Dict, List, Optional

from app.config import settings
from app.jobs.dispatcher import JobDispatcher
from app.jobs.models import JobRecord, JobStatus
from app.jobs.ssh_client import HPCSSHClient

logger = logging.getLogger(__name__)


# SLURM state → our JobStatus mapping
SLURM_STATE_MAP = {
    "PENDING": JobStatus.PENDING,
    "CONFIGURING": JobStatus.PENDING,
    "RUNNING": JobStatus.RUNNING,
    "COMPLETING": JobStatus.RUNNING,
    "COMPLETED": JobStatus.COMPLETED,
    "FAILED": JobStatus.FAILED,
    "TIMEOUT": JobStatus.FAILED,
    "CANCELLED": JobStatus.FAILED,
    "CANCELLED+": JobStatus.FAILED,
    "OUT_OF_MEMORY": JobStatus.FAILED,
    "NODE_FAIL": JobStatus.FAILED,
    "PREEMPTED": JobStatus.FAILED,
}


def _generate_slurm_script(job_id: str, hpc_job_dir: str) -> str:
    """Generate a SLURM batch script for a job."""
    job_id_short = job_id[:8]
    return f"""#!/bin/bash
#SBATCH --job-name=agripay_{job_id_short}
#SBATCH --partition={settings.hpc_slurm_partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={settings.hpc_slurm_cpus}
#SBATCH --mem={settings.hpc_slurm_mem}
#SBATCH --time={settings.hpc_slurm_time_limit}
#SBATCH --output={hpc_job_dir}/slurm_%j.out
#SBATCH --error={hpc_job_dir}/slurm_%j.err

set -eo pipefail

# Force Python stdout unbuffered (prevents buffered output eating SLURM logs)
export PYTHONUNBUFFERED=1

# Initialize module system (not auto-loaded in SLURM batch jobs on this cluster)
source /etc/profile.d/modules.sh 2>/dev/null || true

module purge
module load Miniforge/25.3.1
module load cuda/13.0
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate {settings.hpc_conda_env}

cd {settings.hpc_inference_dir}
python run_inference.py --config {hpc_job_dir}/inputs/config.json

conda deactivate
"""


class HPCDispatcher(JobDispatcher):
    """Dispatches ML inference jobs to the LUMS HPC cluster via SSH/SLURM."""

    def __init__(self, post_process_fn: Callable):
        """
        post_process_fn: callable(job: JobRecord, results_dir: str) -> JobRecord
            Synchronous function that processes HPC results locally
            (loads results.json + heatmap.npz, generates visualizations).
        """
        self._ssh = HPCSSHClient()
        self._post_process_fn = post_process_fn
        self._jobs: Dict[str, JobRecord] = {}
        self._slurm_to_job: Dict[str, str] = {}  # slurm_job_id → job_id
        self._active_slurm_ids: List[str] = []    # currently active SLURM jobs
        self._submit_semaphore: Optional[asyncio.Semaphore] = None
        self._submit_queue: asyncio.Queue[str] = asyncio.Queue()
        self._poll_task: Optional[asyncio.Task] = None
        self._submit_task: Optional[asyncio.Task] = None
        self._running = False
        # Track wall-clock times for timeout detection
        # _job_start_times: when dispatcher submission began (for total-job timeout)
        # _job_pending_times: when SLURM accepted the job (for PENDING-specific timeout)
        self._job_start_times: Dict[str, float] = {}
        self._job_pending_times: Dict[str, float] = {}

    async def submit(self, job: JobRecord) -> str:
        """Submit a job for HPC processing. Returns job_id."""
        job.dispatch_backend = "hpc"
        job.progress_message = "Queued for HPC dispatch..."
        self._jobs[job.id] = job
        await self._submit_queue.put(job.id)
        logger.info("Job %s queued for HPC dispatch", job.id)
        return job.id

    async def get_status(self, job_id: str) -> Optional[JobRecord]:
        """Get current status of a job."""
        return self._jobs.get(job_id)

    async def start(self) -> None:
        """Start the dispatcher: submission loop + polling loop."""
        self._running = True
        self._submit_semaphore = asyncio.Semaphore(settings.hpc_max_concurrent_jobs)
        self._submit_task = asyncio.create_task(self._submission_loop())
        self._poll_task = asyncio.create_task(self._polling_loop())
        logger.info(
            "HPC dispatcher started (max %d concurrent jobs, poll every %ds)",
            settings.hpc_max_concurrent_jobs,
            settings.hpc_poll_interval_seconds,
        )

    async def stop(self) -> None:
        """Stop the dispatcher gracefully."""
        self._running = False
        for task in [self._submit_task, self._poll_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await self._ssh.close()
        logger.info("HPC dispatcher stopped")

    # --- Submission loop ---

    async def _submission_loop(self) -> None:
        """Dequeue jobs and submit them to HPC, respecting concurrency limits."""
        while self._running:
            try:
                job_id = await asyncio.wait_for(self._submit_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # Wait for a slot (max 3 concurrent SLURM jobs)
            await self._submit_semaphore.acquire()

            # Submit in background — don't block the queue for other jobs
            asyncio.create_task(self._submit_job_to_hpc(job_id))

    async def _submit_job_to_hpc(self, job_id: str) -> None:
        """Handle the full submission of a single job to HPC."""
        job = self._jobs.get(job_id)
        if job is None:
            self._submit_semaphore.release()
            return

        try:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.utcnow()

            image_path = job.input_params.get("image_path", "")
            image_filename = os.path.basename(image_path)
            hpc_job_dir = f"{settings.hpc_scratch_dir}/{job_id}"

            # Step 1: Create remote directories
            job.progress_message = "Creating HPC directories..."
            job.progress_current = 2
            job.progress_total = 100
            await self._ssh.mkdir(f"{hpc_job_dir}/inputs")
            await self._ssh.mkdir(f"{hpc_job_dir}/outputs")

            # Step 2: Upload image to HPC
            job.progress_message = "Uploading image to HPC..."
            job.progress_current = 5
            remote_image_path = f"{hpc_job_dir}/inputs/{image_filename}"
            await self._ssh.scp_upload(image_path, remote_image_path)

            # Step 3: Generate and upload config JSON
            job.progress_message = "Preparing HPC job configuration..."
            job.progress_current = 8
            config_json = {
                "job_id": job_id,
                "model_id": job.model_id,
                "image_path": remote_image_path,
                "weights_dir": settings.hpc_models_dir,
                "output_dir": f"{hpc_job_dir}/outputs",
            }
            # Write config to a temp file, upload, then delete temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as tmp:
                json.dump(config_json, tmp)
                tmp_path = tmp.name

            try:
                await self._ssh.scp_upload(tmp_path, f"{hpc_job_dir}/inputs/config.json")
            finally:
                os.unlink(tmp_path)

            # Step 4: Generate and upload SLURM script
            slurm_script = _generate_slurm_script(job_id, hpc_job_dir)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".sh", delete=False
            ) as tmp:
                tmp.write(slurm_script)
                tmp_path = tmp.name

            try:
                await self._ssh.scp_upload(tmp_path, f"{hpc_job_dir}/run.sh")
            finally:
                os.unlink(tmp_path)

            # Step 5: Submit via sbatch
            job.progress_message = "Submitting SLURM job..."
            job.progress_current = 10
            stdout, stderr, exit_code = await self._ssh.run_command(
                f"sbatch {hpc_job_dir}/run.sh",
                timeout=30,
            )

            if exit_code != 0:
                # Check for quota errors
                if "MaxSubmitJobsPerUser" in stderr:
                    raise RuntimeError(
                        "SLURM job quota exceeded (max 3 per user). "
                        "Please wait for running jobs to complete."
                    )
                raise RuntimeError(f"sbatch failed (exit {exit_code}): {stderr.strip()}")

            # Parse SLURM job ID from "Submitted batch job 12345"
            slurm_job_id = stdout.strip().split()[-1]
            job.slurm_job_id = slurm_job_id
            self._slurm_to_job[slurm_job_id] = job_id
            self._active_slurm_ids.append(slurm_job_id)
            self._job_start_times[job_id] = time.time()
            self._job_pending_times[job_id] = time.time()  # start PENDING clock

            job.progress_message = f"SLURM job {slurm_job_id} submitted, waiting in queue..."
            logger.info(
                "Job %s submitted as SLURM job %s", job_id, slurm_job_id
            )

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = f"HPC submission failed: {str(e)}"
            job.completed_at = datetime.utcnow()
            self._submit_semaphore.release()
            logger.error("Job %s HPC submission failed: %s", job_id, str(e))

    # --- Polling loop ---

    async def _polling_loop(self) -> None:
        """Periodically check status of active SLURM jobs."""
        while self._running:
            try:
                await asyncio.sleep(settings.hpc_poll_interval_seconds)

                if not self._active_slurm_ids:
                    continue

                # Check for timeout on all active jobs
                self._check_timeouts()

                # Query SLURM for all active job states in a single SSH call
                slurm_ids_str = ",".join(self._active_slurm_ids)
                stdout, stderr, exit_code = await self._ssh.run_command(
                    f"sacct -j {slurm_ids_str} --format=JobID,State --noheader --parsable2",
                    timeout=15,
                )

                if exit_code != 0:
                    logger.warning("sacct query failed: %s", stderr.strip())
                    continue

                # Parse sacct output
                for line in stdout.strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = line.strip().split("|")
                    if len(parts) < 2:
                        continue

                    sacct_job_id = parts[0].strip()
                    slurm_state = parts[1].strip()

                    # sacct includes sub-job entries like "12345.batch" — skip those
                    if "." in sacct_job_id:
                        continue

                    if sacct_job_id not in self._slurm_to_job:
                        continue

                    job_id = self._slurm_to_job[sacct_job_id]
                    job = self._jobs.get(job_id)
                    if job is None:
                        continue

                    mapped_status = SLURM_STATE_MAP.get(slurm_state)
                    if mapped_status is None:
                        logger.warning(
                            "Unknown SLURM state '%s' for job %s", slurm_state, sacct_job_id
                        )
                        continue

                    # Update progress for running jobs
                    if mapped_status == JobStatus.RUNNING:
                        job.progress_message = "Running on HPC (GPU)..."
                        job.progress_current = 50
                        # Job has started running — reset the pending clock
                        self._job_pending_times.pop(job_id, None)
                    elif mapped_status == JobStatus.PENDING:
                        pending_secs = int(time.time() - self._job_pending_times.get(job_id, time.time()))
                        pending_hrs = pending_secs // 3600
                        pending_mins = (pending_secs % 3600) // 60
                        job.progress_message = f"SLURM job {sacct_job_id} queued ({pending_hrs}h {pending_mins}m)..."
                        job.progress_current = 10

                    # Handle completion
                    if mapped_status == JobStatus.COMPLETED:
                        asyncio.create_task(
                            self._handle_job_completed(job_id, sacct_job_id)
                        )

                    # Handle failure
                    elif mapped_status == JobStatus.FAILED:
                        error_detail = self._slurm_error_detail(slurm_state)
                        asyncio.create_task(
                            self._handle_job_failed(job_id, sacct_job_id, error_detail)
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Polling loop error: %s", str(e))

    def _check_timeouts(self) -> None:
        """Check if any active jobs have exceeded their timeout.

        Two independent timeouts apply:
        - hpc_job_timeout_seconds: total wall-clock time from dispatcher submission.
          Guards against jobs that run but never complete.
        - hpc_slurm_pending_timeout_seconds: time spent in SLURM PENDING state.
          Guards against jobs stuck in queue indefinitely (other users, node issues).
          Defaults to 86400s (24 hours).
        """
        now = time.time()
        total_timeout = settings.hpc_job_timeout_seconds
        pending_timeout = settings.hpc_slurm_pending_timeout_seconds
        timed_out = []

        for slurm_id in self._active_slurm_ids:
            job_id = self._slurm_to_job.get(slurm_id)
            if not job_id:
                continue

            # Check total job timeout
            if job_id in self._job_start_times:
                total_elapsed = now - self._job_start_times[job_id]
                if total_elapsed > total_timeout:
                    timed_out.append((slurm_id, job_id, f"Job timed out after {total_timeout}s total"))
                    continue

            # Check PENDING-specific timeout (only if still in pending clock)
            if job_id in self._job_pending_times:
                pending_elapsed = now - self._job_pending_times[job_id]
                if pending_elapsed > pending_timeout:
                    pending_hrs = int(pending_elapsed // 3600)
                    timed_out.append((
                        slurm_id, job_id,
                        f"Job stuck in SLURM PENDING for {pending_hrs}h — cancelled (max {pending_timeout//3600}h)",
                    ))

        for slurm_id, job_id, reason in timed_out:
            logger.warning("Job %s (SLURM %s): %s", job_id, slurm_id, reason)
            asyncio.create_task(self._handle_timeout(job_id, slurm_id, reason))

    async def _handle_timeout(self, job_id: str, slurm_job_id: str, reason: str = "") -> None:
        """Cancel a timed-out SLURM job and mark it failed."""
        try:
            await self._ssh.run_command(f"scancel {slurm_job_id}", timeout=10)
        except Exception as e:
            logger.warning("Failed to scancel %s: %s", slurm_job_id, str(e))

        await self._handle_job_failed(
            job_id, slurm_job_id,
            reason or f"Job timed out after {settings.hpc_job_timeout_seconds}s",
        )

    @staticmethod
    def _slurm_error_detail(slurm_state: str) -> str:
        """Generate a human-readable error message from a SLURM failure state."""
        details = {
            "FAILED": "SLURM job failed (check inference logs)",
            "TIMEOUT": "HPC job exceeded SLURM time limit",
            "CANCELLED": "SLURM job was cancelled",
            "CANCELLED+": "SLURM job was cancelled",
            "OUT_OF_MEMORY": "HPC job ran out of memory (OOM)",
            "NODE_FAIL": "HPC node failure",
            "PREEMPTED": "SLURM job was preempted by a higher-priority job",
        }
        return details.get(slurm_state, f"SLURM job ended with state: {slurm_state}")

    # --- Completion handling ---

    async def _handle_job_completed(self, job_id: str, slurm_job_id: str) -> None:
        """Download results from HPC, run local post-processing, cleanup."""
        job = self._jobs.get(job_id)
        if job is None:
            self._cleanup_tracking(slurm_job_id, job_id)
            return

        hpc_job_dir = f"{settings.hpc_scratch_dir}/{job_id}"

        try:
            # Step 1: Check status.json from HPC
            job.progress_message = "Downloading results from HPC..."
            job.progress_current = 85

            # Create local temp dir for downloaded results
            local_results_dir = tempfile.mkdtemp(prefix=f"hpc_results_{job_id[:8]}_")

            # Download status.json first
            try:
                await self._ssh.scp_download(
                    f"{hpc_job_dir}/outputs/status.json",
                    os.path.join(local_results_dir, "status.json"),
                )
                with open(os.path.join(local_results_dir, "status.json")) as f:
                    status_data = json.load(f)

                if status_data.get("status") == "failed":
                    error_msg = status_data.get("error", "Unknown inference error")
                    await self._handle_job_failed(
                        job_id, slurm_job_id, f"Inference failed: {error_msg}"
                    )
                    return
            except Exception as e:
                logger.warning("Could not read status.json: %s", str(e))
                # Continue — results.json might still be there

            # Step 2: Download results.json
            await self._ssh.scp_download(
                f"{hpc_job_dir}/outputs/results.json",
                os.path.join(local_results_dir, "results.json"),
            )

            # Step 3: Download heatmap.npz
            try:
                await self._ssh.scp_download(
                    f"{hpc_job_dir}/outputs/heatmap.npz",
                    os.path.join(local_results_dir, "heatmap.npz"),
                )
            except Exception as e:
                logger.warning("Heatmap download failed (non-fatal): %s", str(e))

            # Step 4: Run local post-processing (visualizations)
            job.progress_message = "Generating visualizations..."
            job.progress_current = 95

            loop = asyncio.get_event_loop()
            updated_job = await loop.run_in_executor(
                None, self._post_process_fn, job, local_results_dir
            )
            self._jobs[job_id] = updated_job

            logger.info("Job %s completed successfully via HPC", job_id)

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = f"Post-processing failed: {str(e)}\n{traceback.format_exc()}"
            job.completed_at = datetime.utcnow()
            logger.error("Job %s post-processing failed: %s", job_id, str(e))

        finally:
            # Step 5: Cleanup HPC scratch directory
            try:
                await self._ssh.rm_rf(hpc_job_dir)
            except Exception as e:
                logger.warning("HPC cleanup failed for %s: %s", job_id, str(e))

            self._job_pending_times.pop(job_id, None)
            self._cleanup_tracking(slurm_job_id, job_id)

    async def _handle_job_failed(
        self, job_id: str, slurm_job_id: str, error_detail: str
    ) -> None:
        """Mark a job as failed, try to download error logs, cleanup."""
        job = self._jobs.get(job_id)
        if job is None:
            self._cleanup_tracking(slurm_job_id, job_id)
            return

        # Try to download SLURM error log for diagnostics
        hpc_job_dir = f"{settings.hpc_scratch_dir}/{job_id}"
        try:
            stdout, _, _ = await self._ssh.run_command(
                f"cat {hpc_job_dir}/slurm_*.err 2>/dev/null | tail -50",
                timeout=10,
            )
            if stdout.strip():
                error_detail += f"\n\nSLURM stderr (last 50 lines):\n{stdout.strip()}"
        except Exception:
            pass  # Best-effort log retrieval

        job.status = JobStatus.FAILED
        job.error = error_detail
        job.completed_at = datetime.utcnow()
        logger.error("Job %s failed: %s", job_id, error_detail[:200])

        # Cleanup HPC scratch
        try:
            await self._ssh.rm_rf(hpc_job_dir)
        except Exception:
            pass

        self._job_pending_times.pop(job_id, None)
        self._cleanup_tracking(slurm_job_id, job_id)

    def _cleanup_tracking(self, slurm_job_id: str, job_id: str) -> None:
        """Remove a job from active tracking and release semaphore slot."""
        if slurm_job_id in self._active_slurm_ids:
            self._active_slurm_ids.remove(slurm_job_id)
        self._slurm_to_job.pop(slurm_job_id, None)
        self._job_start_times.pop(job_id, None)
        if self._submit_semaphore:
            self._submit_semaphore.release()
