"""Blockchain oracle — releases escrowed USDT after milestone verification.

Ported from backend_blockchain/oracle.js into the compute-backend as a FastAPI
router so everything runs on the same container and port.

Env vars required:
  ORACLE_PRIVATE_KEY        — Sepolia wallet key that holds ORACLE_ROLE
  ESCROW_CONTRACT_ADDRESS   — deployed CropEscrow proxy
  RPC_URL                   — Sepolia JSON-RPC (default: public node)
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from web3 import Web3

from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/oracle", tags=["oracle"])

# ── ABI (only the function we call) ──────────────────────────────────────────
RELEASE_ABI = [
    {
        "inputs": [{"internalType": "bytes32", "name": "_milestoneId", "type": "bytes32"}],
        "name": "releasePayment",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    }
]

# ── Web3 setup (lazy — tolerates missing credentials at import time) ─────────
_w3 = None
_escrow = None
_account = None


def _get_web3():
    """Lazily initialise Web3 + contract so the server can still start without
    oracle env vars (other endpoints remain available)."""
    global _w3, _escrow, _account

    if _w3 is not None:
        return _w3, _escrow, _account

    rpc_url = settings.oracle_rpc_url
    pk = settings.oracle_private_key
    addr = settings.escrow_contract_address

    if not pk or not addr:
        logger.warning("Oracle env vars missing — running in MOCK mode")
        return None, None, None

    _w3 = Web3(Web3.HTTPProvider(rpc_url))
    _account = _w3.eth.account.from_key(pk)
    _escrow = _w3.eth.contract(address=Web3.to_checksum_address(addr), abi=RELEASE_ABI)
    logger.info("Oracle connected as %s on %s", _account.address, rpc_url)
    return _w3, _escrow, _account


# ── Supabase helpers (reuse the SDK already in requirements) ─────────────────
def _sb():
    """Return the service-role Supabase client (sync)."""
    from supabase import create_client

    return create_client(settings.supabase_url, settings.supabase_service_role_key)


# ── Request / Response models ────────────────────────────────────────────────
class ReleaseRequest(BaseModel):
    milestone_id: str


class ReleaseResponse(BaseModel):
    status: str
    tx_hash: str | None = None
    block: int | None = None
    mock: bool = False


# ── POST /api/v1/oracle/release ──────────────────────────────────────────────
@router.post("/release", response_model=ReleaseResponse)
async def release_payment(body: ReleaseRequest):
    """Call CropEscrow.releasePayment on-chain, then update Supabase."""
    milestone_id = body.milestone_id
    if not milestone_id:
        raise HTTPException(status_code=400, detail="Missing milestone_id")

    w3, escrow, account = _get_web3()

    # ── Mock mode ────────────────────────────────────────────────────────────
    if escrow is None:
        logger.info("[MOCK] Releasing funds for milestone %s", milestone_id)
        return ReleaseResponse(status="success", tx_hash="0xMOCK", mock=True)

    # ── On-chain release ─────────────────────────────────────────────────────
    try:
        milestone_bytes = Web3.solidity_keccak(["string"], [milestone_id])

        tx = escrow.functions.releasePayment(milestone_bytes).build_transaction(
            {
                "from": account.address,
                "nonce": w3.eth.get_transaction_count(account.address),
                "gas": 200_000,
                "gasPrice": w3.eth.gas_price,
            }
        )
        signed = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        logger.info("Oracle tx broadcast: %s", tx_hash.hex())

        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        logger.info("Confirmed in block %d", receipt.blockNumber)
    except Exception as exc:
        logger.error("On-chain release failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Blockchain tx failed: {exc}")

    # ── Update Supabase ──────────────────────────────────────────────────────
    try:
        sb = _sb()

        # 1. Mark milestone as paid
        sb.table("cycle_milestones").update(
            {"payment_status": "paid", "payment_released_at": "now()"}
        ).eq("id", milestone_id).execute()

        # 2. Record release transaction
        sb.table("transactions").insert(
            {
                "cycle_milestone_id": milestone_id,
                "tx_hash": receipt.transactionHash.hex(),
                "amount": 0,  # release tx has no transfer amount on escrow side
                "status": "confirmed",
                "blockchain_network": "sepolia",
                "from_address": account.address,
                "to_address": settings.escrow_contract_address,
                "gas_fee_wei": receipt.gasUsed * receipt.effectiveGasPrice,
            }
        ).execute()

        # 3. Traverse to farmer for notification
        ms_row = (
            sb.table("cycle_milestones")
            .select("crop_cycle_id")
            .eq("id", milestone_id)
            .single()
            .execute()
        )
        cycle_row = (
            sb.table("crop_cycles")
            .select("farm_id")
            .eq("id", ms_row.data["crop_cycle_id"])
            .single()
            .execute()
        )
        farm_row = (
            sb.table("farms")
            .select("user_id, name")
            .eq("id", cycle_row.data["farm_id"])
            .single()
            .execute()
        )
        if farm_row.data.get("user_id"):
            sb.table("notifications").insert(
                {
                    "user_id": farm_row.data["user_id"],
                    "message": (
                        f'Payment received! A milestone for farm '
                        f'"{farm_row.data["name"]}" has been released to your wallet.'
                    ),
                    "is_read": False,
                    "link_to": f'/payments/{cycle_row.data["farm_id"]}',
                }
            ).execute()

        logger.info("Milestone %s marked paid, farmer notified.", milestone_id)
    except Exception as db_err:
        # Contract tx succeeded; DB failure is non-fatal
        logger.error("Supabase update failed (tx was OK): %s", db_err)

    return ReleaseResponse(
        status="success",
        tx_hash=receipt.transactionHash.hex(),
        block=receipt.blockNumber,
    )


# ── GET /api/v1/oracle/health ────────────────────────────────────────────────
@router.get("/health")
async def oracle_health():
    _, escrow, account = _get_web3()
    return {
        "status": "healthy",
        "mode": "active" if escrow else "mock",
        "address": account.address if account else None,
    }
