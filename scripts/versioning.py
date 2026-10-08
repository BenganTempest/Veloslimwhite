"""
Model versioning: which scoring method produced which numbers.

The version label comes from config.MODEL_VERSION, but the pipeline doesn't
just trust it -- it fingerprints every setting that changes how scores are
calculated. If the fingerprint changes while MODEL_VERSION stays the same
(someone tweaked a weight and forgot to bump the version), an automatic
sub-version like "2.0+3fa1c9" is registered and a warning appears in the
Actions log, so results are never silently mixed across methods.

The register (data/model_versions.json) records each version's first run
date, note and exact settings -- the audit trail for "what changed when".
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# Every config value that changes how a score is CALCULATED (not how it's
# displayed or alerted on). Add to this list if you add a new scoring knob.
SCORING_KEYS = [
    "PRICE_LOOKBACK", "MOMENTUM_WINDOWS", "VOLATILITY_WINDOW", "RSI_WINDOW",
    "VOLUME_SPIKE_SHORT", "VOLUME_SPIKE_LONG", "BREAKOUT_RETURN", "BREAKOUT_WINDOW",
    "VALIDATION_HOLDOUT_DAYS", "WEIGHT_PATTERN", "WEIGHT_MOMENTUM", "WEIGHT_SENTIMENT",
    "MOMENTUM_SECTOR_WEIGHT", "MOMENTUM_MIN_SECTOR_SIZE", "LIQUIDITY_WINDOW", "MIN_AVG_TURNOVER",
    "FETCH_NEWS", "NEWS_LOOKBACK_DAYS", "INCLUDE_SP500", "INCLUDE_NASDAQ100", "INCLUDE_OMX_STOCKHOLM_ALL",
]


def scoring_settings() -> dict:
    return {k: getattr(config, k, None) for k in SCORING_KEYS}


def fingerprint(settings: dict | None = None) -> str:
    blob = json.dumps(settings or scoring_settings(), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def load_register() -> list[dict]:
    path = config.MODEL_VERSIONS_FILE
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text()).get("versions", [])
    except Exception:  # noqa: BLE001
        return []


def save_register(versions: list[dict]) -> None:
    path = config.MODEL_VERSIONS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"versions": versions}, indent=2, ensure_ascii=False) + "\n")


def resolve_version(today: str, register: list[dict] | None = None, persist: bool = True) -> tuple[str, list[dict]]:
    """
    Returns (version_label, register). Registers a new version on first use.
    Note: the weights used are the CONFIGURED ones -- a run where the pattern
    model couldn't be trained still counts as the same version (the run log
    records model_trained separately).
    """
    versions = list(register if register is not None else load_register())
    fp = fingerprint()
    for v in versions:
        if v.get("fingerprint") == fp:
            return v["version"], versions

    label = config.MODEL_VERSION
    if any(v["version"] == label for v in versions):
        # same label, different settings: auto sub-version + visible warning
        label = f"{config.MODEL_VERSION}+{fp[:6]}"
        print(f"::warning title=Modellversion::Poängberäkningen har ändrats men MODEL_VERSION är "
              f"fortfarande {config.MODEL_VERSION}. Registrerar automatiskt {label}. "
              f"Höj MODEL_VERSION och skriv en kort MODEL_VERSION_NOTE i config.py.", flush=True)
        note = "Automatisk version: inställningar ändrade utan att MODEL_VERSION höjdes."
    else:
        note = config.MODEL_VERSION_NOTE
        print(f"::notice title=Modellversion::Ny modellversion {label} registrerad.", flush=True)

    versions.append({
        "version": label,
        "fingerprint": fp,
        "first_date": today,
        "note": note,
        "settings": scoring_settings(),
    })
    if persist:
        save_register(versions)
    return label, versions


def with_legacy(versions: list[dict], history_versions: list[str]) -> list[dict]:
    """Adds a register entry for the pre-versioning label if old snapshots exist."""
    if config.LEGACY_MODEL_VERSION in history_versions and not any(
        v["version"] == config.LEGACY_MODEL_VERSION for v in versions
    ):
        return [{"version": config.LEGACY_MODEL_VERSION, "fingerprint": None, "first_date": None,
                 "note": "Ursprunglig modell, innan versionshantering infördes.", "settings": None}] + versions
    return versions
