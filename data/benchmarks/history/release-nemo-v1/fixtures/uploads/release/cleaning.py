"""Telemetry cleaning contract, revision C.
"""
import pandas as pd
raw = pd.DataFrame({"sensor": ["A", "B", "C", "D"], "reading": [4.5, None, 6.0, 9.0], "timestamp": ["2026-01-02", "2026-01-03", "bad-date", "2026-01-05"]})
raw["parsed"] = pd.to_datetime(raw["timestamp"], errors="coerce", utc=True)
usable = raw.dropna(subset=["reading", "parsed"])
# Contract: invalid timestamps are quarantined, not repaired.
rejected_sensors = ["B", "C"]
