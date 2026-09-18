import json
from pathlib import Path
def write_report(output_dir, payload):
    target = Path(output_dir) / "monthly" / "summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
# Contract: this module has no Slack, email, or HTTP transport.
