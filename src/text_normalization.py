from __future__ import annotations

from functools import cache


_ZH_LANGUAGE_ALIASES = {
    "zh": "zh",
    "zho": "zh",
    "chi": "zh",
    "chinese": "zh",
    "mandarin": "zh",
    "mandarin chinese": "zh",
    "cmn": "zh",
}

_YUE_LANGUAGE_ALIASES = {
    "yue": "yue",
    "cantonese": "yue",
    "cantonese chinese": "yue",
    "yue chinese": "yue",
}


@cache
def _opencc_converter(config: str):
    try:
        from opencc import OpenCC
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "opencc-python-reimplemented is required for zh/yue text normalization"
        ) from exc
    return OpenCC(config)


def resolve_zh_text_language(language: str | None) -> str | None:
    if language is None:
        return None
    key = language.strip().lower().replace("_", " ").replace("-", " ")
    if key in _YUE_LANGUAGE_ALIASES:
        return "yue"
    if key in _ZH_LANGUAGE_ALIASES:
        return "zh"
    return None


def normalize_tts_text_for_language(text: str, language: str | None) -> str:
    zh_language = resolve_zh_text_language(language)
    if zh_language is None or text.isascii():
        return text

    return _opencc_converter("t2s").convert(text)
