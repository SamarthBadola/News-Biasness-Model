"""Write the API's OpenAPI document to api/openapi.json. Run after changing newsbias/api/schema.py.

    python scripts/export_openapi.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from newsbias.api.server import app  # noqa: E402

out = ROOT / "api" / "openapi.json"
out.write_text(json.dumps(app.openapi(), indent=2) + "\n", encoding="utf-8")
print(f"wrote {out}")
