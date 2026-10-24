"""Central settings for the MuleGuard data-prep pipeline.

Everything tunable lives here. Override the data folder with the
MULEGUARD_DATA environment variable if you keep data elsewhere.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("MULEGUARD_DATA", ROOT / "data"))
RAW = DATA / "raw"
CLEAN = DATA / "clean"
FINAL = DATA / "final"

RAW_TRANS = RAW / "HI-Small_Trans.csv"
RAW_PATTERNS = RAW / "HI-Small_Patterns.txt"


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


SEED = _int("MULEGUARD_SEED", 42)

# ---------------------------------------------------------------- injection
# ~1,500 scenarios -> ~2,700 mule accounts (~0.5% of HI-Small's ~515K accounts), enough to train on.
N_MULE_SCENARIOS = _int("MULEGUARD_N_MULES", 1500)
N_DECOY_SCENARIOS = _int("MULEGUARD_N_DECOYS", 1000)

# Inbound amount of an injected mule scenario, in rupees (inclusive).
MULE_AMOUNT = (300_000, 800_000)

# single_pass: 1 outbound | fan_out: 3-10 outbound | two_hop: 1-3 recipients that forward again
MULE_KIND_WEIGHTS = {"single_pass": 0.30, "fan_out": 0.30, "two_hop": 0.40}

# Dwell time = seconds between money arriving and the first outbound leaving.
# Most mules are "fast"; a minority are "medium" so the model cannot just learn "< 2 min".
FAST_PROFILE_PROB = 0.85
FAST_DWELL_S = (20, 120)
MEDIUM_DWELL_S = (120, 600)
HOP2_DELAY_S = (60, 300)          # second-layer mule forwards 1-5 min after receiving
FORWARD_FRACTION = (0.90, 0.99)   # share of the inbound amount that is passed on

DECOY_KIND_WEIGHTS = {
    "biz_supplier": 0.35,         # business gets paid, pays suppliers hours/days later
    "aggregator": 0.15,           # many small payments in, one daily settlement out
    "one_off": 0.35,              # one big personal transfer, no fast forwarding
    "trusted_forwarder": 0.15,    # repeat fast pass-through between the SAME two counterparties
}

# Accounts with at most this many base transactions are eligible to become mules / decoys.
LOW_ACTIVITY_MAX = 10
