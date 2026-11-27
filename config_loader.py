import json
from pathlib import Path
LANGUAGES = json.loads(Path("config/languages.json").read_text(encoding="utf-8"))
