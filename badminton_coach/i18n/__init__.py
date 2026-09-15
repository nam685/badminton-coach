"""Small static-string translations for UI chrome burned into the video (HUD/banners) and the
report/index.html (spec §6.6). Judge-generated text (findings, cues, drills) is already in the target
language via the judge's own LANGUAGE instruction — this module is only for labels *we* draw/render.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_DIR = Path(__file__).resolve().parent
SUPPORTED_LANGS = ("vi", "de", "en")


@lru_cache(maxsize=len(SUPPORTED_LANGS))
def _load(lang: str) -> dict[str, str]:
    path = _DIR / f"{lang}.json"
    if not path.exists():
        path = _DIR / "en.json"
    return json.loads(path.read_text())


def t(key: str, lang: str = "en") -> str:
    """Looks up `key` in `lang`'s strings, falling back to English, then to `key` itself."""
    strings = _load(lang if lang in SUPPORTED_LANGS else "en")
    if key in strings:
        return strings[key]
    return _load("en").get(key, key)
