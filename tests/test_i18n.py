from __future__ import annotations

import json
from pathlib import Path

from badminton_coach.i18n import SUPPORTED_LANGS, t

_I18N_DIR = Path(__file__).resolve().parent.parent / "badminton_coach" / "i18n"


def test_all_lang_files_have_identical_key_sets() -> None:
    key_sets = {}
    for lang in SUPPORTED_LANGS:
        data = json.loads((_I18N_DIR / f"{lang}.json").read_text())
        key_sets[lang] = set(data.keys())
    reference = key_sets["en"]
    for lang, keys in key_sets.items():
        assert keys == reference, f"{lang}.json key set differs from en.json: {keys.symmetric_difference(reference)}"


def test_t_returns_correct_language() -> None:
    assert t("swing", "en") == "swing"
    assert t("swing", "vi") != "swing"
    assert t("swing", "de") != "swing"


def test_t_falls_back_to_english_for_unknown_lang() -> None:
    assert t("swing", "fr") == t("swing", "en")


def test_t_falls_back_to_key_for_unknown_key() -> None:
    assert t("this_key_does_not_exist", "en") == "this_key_does_not_exist"
