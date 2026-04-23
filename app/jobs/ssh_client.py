"""Async SSH/SCP client for HPC communication.

Uses asyncssh for async-native SSH (matching FastAPI's async architecture)
with password authentication per HPC manual convention.

For large file transfers (multi-GB GeoTIFFs), falls back to subprocess-based
sshpass+scp to avoid loading entire files into Python memory.
"""

import asyncio
import logging
import os
import shutil
from typing import Optional, Tuple

import asyncssh

from app.config import settings

logger = logging.getLogger(__name__)

# Threshold for using subprocess SCP vs asyncssh SCP (100 MB)
LARGE_FILE_THRESHOLD_BYTES = 100 * 1024 * 1024


class HPCSSHClient:
    """Manages SSH/SCP connections to the HPC cluster."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self._host = host or settings.hpc_ssh_host
        self._port = port or settings.hpc_ssh_port
        self._username = username or settings.hpc_ssh_user
        self._password = password or settings.hpc_ssh_password
        self._timeout = timeout or settings.hpc_ssh_timeout_seconds
        self._conn: Optional[asyncssh.SSHClientConnection] = None

    async def connect(self) -> None:
        """Establish SSH connection to HPC. Reconnects if already connected."""
        if self._conn is not None:
            try:
                # Check if connection is still alive
                await self._conn.run("echo ok", timeout=5)
                return  # Connection is good
            except Exception:
                # Connection is dead, close and reconnect
                await self.close()

        logger.info("Connecting to HPC %s@%s:%d", self._username, self._host, self._port)
        self._conn = await asyncssh.connect(
            host=self._host,
            port=self._port,
            username=self._username,
            password=self._password,
            known_hosts=None,  # Skip host key verification (university internal network)
            connect_timeout=self._timeout,
        )
        logger.info("SSH connection established to %s", self._host)

    async def close(self) -> None:
        """Close the SSH connection."""
        if self._conn is not None:
            self._conn.close()
            await self._conn.wait_closed()
            self._conn = None
            logger.info("SSH connection closed")

    async def run_command(
        self, cmd: str, timeout: Optional[int] = None
    ) -> Tuple[str, str, int]:
        """Execute a command on the HPC via SSH.

        Returns:
            Tuple of (stdout, stderr, exit_code).
            exit_code is -1 if the command timed out or connection failed.
        """
        await self.connect()
        effective_timeout = timeout or self._timeout

        try:
            result = await self._conn.run(cmd, timeout=effective_timeout)
            exit_code = result.exit_status if result.exit_status is not None else -1
            stdout = result.stdout or ""
            stderr = result.stderr or ""

            if exit_code != 0:
                logger.warning(
                    "Command failed (exit %d): %s\nstderr: %s",
                    exit_code, cmd, stderr.strip()
                )
            return stdout, stderr, exit_code

        except asyncssh.TimeoutError:
            logger.error("SSH command timed out after %ds: %s", effective_timeout, cmd)
            return "", f"Command timed out after {effective_timeout}s", -1
        except Exception as e:
            logger.error("SSH command failed: %s — %s", cmd, str(e))
            # Connection might be dead
            await self.close()
            raise

    async def mkdir(self, remote_path: str) -> None:
        """Create a directory on the HPC (mkdir -p)."""
        stdout, stderr, exit_code = await self.run_command(f"mkdir -p {remote_path}")
        if exit_code != 0:
            raise RuntimeError(f"Failed to create directory {remote_path}: {stderr}")

    async def rm_rf(self, remote_path: str) -> None:
        """Remove a directory/file on the HPC (rm -rf)."""
        stdout, stderr, exit_code = await self.run_command(f"rm -rf {remote_path}")
        if exit_code != 0:
            logger.warning("Failed to remove %s: %s", remote_path, stderr)

    async def file_exists(self, remote_path: str) -> bool:
        """Check if a file exists on the HPC."""
        _, _, exit_code = await self.run_command(f"test -f {remote_path}")
        return exit_code == 0

    async def scp_upload(self, local_path: str, remote_path: str) -> None:
        """Upload a file to the HPC.

        Uses subprocess scp+sshpass for large files (avoids loading into memory).
        Uses asyncssh.scp for small files (simpler, no extra dependencies).
        """
        file_size = os.path.getsize(local_path)
        logger.info(
            "Uploading %s (%d MB) -> %s:%s",
            local_path, file_size // (1024 * 1024), self._host, remote_path,
        )

        if file_size > LARGE_FILE_THRESHOLD_BYTES and shutil.which("sshpass"):
            await self._scp_upload_subprocess(local_path, remote_path)
        else:
            await self._scp_upload_asyncssh(local_path, remote_path)

        logger.info("Upload complete: %s", remote_path)

    async def scp_download(self, remote_path: str, local_path: str) -> None:
        """Download a file from the HPC.

        Uses subprocess sshpass+scp when available to stream directly to disk
        (avoids loading the file into Python/asyncssh memory — critical for
        large outputs like heatmap.npz which can be several hundred MB).
        Falls back to asyncssh when sshpass is not installed.
        """
        logger.info(
            "Downloading %s:%s -> %s",
            self._host, remote_path, local_path,
        )

        if shutil.which("sshpass"):
            await self._scp_download_subprocess(remote_path, local_path)
        else:
            await self._scp_download_asyncssh(remote_path, local_path)

        logger.info("Download complete: %s", local_path)

    async def scp_download_large(self, remote_path: str, local_path: str) -> None:
        """Download a large file from HPC using subprocess scp."""
        if shutil.which("sshpass"):
            await self._scp_download_subprocess(remote_path, local_path)
        else:
            await self._scp_download_asyncssh(remote_path, local_path)

    # --- Internal methods ---

    async def _scp_upload_asyncssh(self, local_path: str, remote_path: str) -> None:
        """Upload using asyncssh.scp (loads file into memory)."""
        await self.connect()
        await asyncssh.scp(
            local_path,
            (self._conn, remote_path),
        )

    async def _scp_download_asyncssh(self, remote_path: str, local_path: str) -> None:
        """Download using asyncssh.scp."""
        await self.connect()
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        await asyncssh.scp(
            (self._conn, remote_path),
            local_path,
        )

    async def _scp_upload_subprocess(self, local_path: str, remote_path: str) -> None:
        """Upload using subprocess sshpass+scp for large files.

        This avoids loading multi-GB files into Python memory.
        Uses SSHPASS env var (not -p flag) to keep password out of the process list.
        """
        remote_target = f"{self._username}@{self._host}:{remote_path}"
        cmd = [
            "sshpass", "-e",
            "scp",
            "-P", str(self._port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            local_path,
            remote_target,
        ]
        env = {**os.environ, "SSHPASS": self._password}

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode().strip()
            raise RuntimeError(
                f"SCP upload failed (exit {proc.returncode}): {error_msg}"
            )

    async def _scp_download_subprocess(self, remote_path: str, local_path: str) -> None:
        """Download using subprocess sshpass+scp.

        Streams directly to disk — safe for large files (e.g. 364MB heatmap.npz).
        Uses SSHPASS env var to keep password out of the process list.
        """
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        remote_source = f"{self._username}@{self._host}:{remote_path}"
        cmd = [
            "sshpass", "-e",
            "scp",
            "-P", str(self._port),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            remote_source,
            local_path,
        ]
        env = {**os.environ, "SSHPASS": self._password}

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            error_msg = stderr.decode().strip()
            raise RuntimeError(
                f"SCP download failed (exit {proc.returncode}): {error_msg}"
            )

    async def is_connected(self) -> bool:
        """Check if SSH connection is alive."""
        if self._conn is None:
            return False
        try:
            await self._conn.run("echo ok", timeout=5)
            return True
        except Exception:
            return False
