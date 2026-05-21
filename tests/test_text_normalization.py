from __future__ import annotations

from text_normalization import normalize_tts_text_for_language, resolve_zh_text_language


def test_resolve_zh_text_language() -> None:
    assert resolve_zh_text_language("zh") == "zh"
    assert resolve_zh_text_language("Chinese") == "zh"
    assert resolve_zh_text_language("yue") == "yue"
    assert resolve_zh_text_language("Cantonese") == "yue"
    assert resolve_zh_text_language("en") is None


def test_yue_and_zh_use_t2s_only() -> None:
    text = "銜接香港與深圳，係唔係啱？"
    expected = "衔接香港与深圳，系唔系啱？"

    assert normalize_tts_text_for_language(text, "yue") == expected
    assert normalize_tts_text_for_language(text, "zh") == expected


def test_non_chinese_language_is_unchanged() -> None:
    text = "銜接香港與深圳"
    assert normalize_tts_text_for_language(text, "en") == text
