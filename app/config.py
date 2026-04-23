"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Supabase
    supabase_url: str = ""
    supabase_service_role_key: str = ""
    supabase_anon_key: str = ""

    # Image storage
    imagery_dir: str = "/data"
    model_weights_dir: str = "/app/model_weights"

    # Compute dispatch
    compute_dispatch_mode: str = "local"  # "local", "hpc", or "auto"
    compute_port: int = 8001

    # HPC (Rustam2) — only used when compute_dispatch_mode is "hpc" or "auto"
    hpc_ssh_host: str = "203.135.63.78"
    hpc_ssh_port: int = 22
    hpc_ssh_user: str = ""
    hpc_ssh_password: str = ""
    hpc_base_dir: str = "~/agripay"
    hpc_models_dir: str = "~/agripay/models"
    hpc_inference_dir: str = "~/agripay/inference"
    hpc_scratch_dir: str = "/scratch/muhammadtalha/agripay"
    hpc_conda_env: str = "agripay"
    hpc_slurm_partition: str = "debug"
    hpc_slurm_time_limit: str = "01:00:00"   # 1 hour: covers first-run CUDA kernel compile (~10min) + inference
    hpc_slurm_cpus: int = 8
    hpc_slurm_mem: str = "48G"
    hpc_max_concurrent_jobs: int = 3
    hpc_poll_interval_seconds: int = 10
    hpc_job_timeout_seconds: int = 7200          # 2hr total wall-clock (submission → completion)
    hpc_slurm_pending_timeout_seconds: int = 86400  # 24hr max wait in SLURM PENDING queue
    hpc_ssh_timeout_seconds: int = 30

    # CORS — comma-separated origins, empty = use defaults
    cors_origins: str = ""

    # Model registry
    max_loaded_models: int = 3

    # Job processing
    job_result_ttl_hours: int = 2
    max_image_dimension: int = 50000

    # Max upload size (bytes) for drone imagery
    max_upload_bytes: int = 10 * 1024 * 1024 * 1024  # 10 GB

    # Google Drive API key (server-side, for Drive → server downloads)
    google_drive_api_key: str = ""

    # Modal (only when dispatch_mode=modal)
    modal_token_id: Optional[str] = None
    modal_token_secret: Optional[str] = None

    # Blockchain oracle (Sepolia escrow release)
    oracle_private_key: str = ""
    escrow_contract_address: str = ""
    oracle_rpc_url: str = "https://ethereum-sepolia-rpc.publicnode.com"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
