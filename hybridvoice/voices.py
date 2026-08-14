"""
Маппинг языков на голоса (Voice Design).

Соответствует логике Dublaj-клиента: для каждого языка по умолчанию
используется мужской голос. Ключ с суффиксом "_female" даёт женский голос.

Значения — это instruct-строки для OmniVoice Voice Design режима.
"""

from typing import Optional


# ============================================================
# Маппинг язык → instruct-голос (по умолчанию = мужской)
# ============================================================
# Ключ "xx"        → мужской голос для языка xx
# Ключ "xx_female" → женский голос для языка xx
# ============================================================
LANGUAGE_VOICE_MAP = {
    # Русский
    "ru": "male",
    "ru_female": "female",
    # Английский
    "en": "male",
    "en_female": "female",
    # Турецкий
    "tr": "male",
    "tr_female": "female",
    # Немецкий
    "de": "male",
    "de_female": "female",
    # Французский
    "fr": "male",
    "fr_female": "female",
    # Испанский
    "es": "male",
    "es_female": "female",
    # Итальянский
    "it": "male",
    "it_female": "female",
    # Португальский
    "pt": "male",
    "pt_female": "female",
    # Польский
    "pl": "male",
    "pl_female": "female",
    # Украинский
    "uk": "male",
    "uk_female": "female",
    # Арабский
    "ar": "male",
    "ar_female": "female",
    # Китайский
    "zh": "male",
    "zh_female": "female",
    # Японский
    "ja": "male",
    "ja_female": "female",
    # Корейский
    "ko": "male",
    "ko_female": "female",
    # Хинди
    "hi": "male",
    "hi_female": "female",
    # Нидерландский
    "nl": "male",
    "nl_female": "female",
    # Шведский
    "sv": "male",
    "sv_female": "female",
    # Вьетнамский
    "vi": "male",
    "vi_female": "female",
}


def get_voice_for_language(
    language: Optional[str],
    female: bool = False,
    default_voice: str = "male",
) -> str:
    """
    Возвращает instruct-голос для указанного языка.

    Логика как в Dublaj: для языка "ru" вернёт "male",
    для "ru" + female=True вернёт "female".
    Если язык не найден в маппинге — вернёт default_voice.

    Args:
        language: Код языка ISO 639 (например, "ru", "en", "tr").
            Если None — используется default_voice.
        female: Если True — вернуть женский голос.
        default_voice: Голос по умолчанию, если язык не найден.

    Returns:
        Instruct-строка голоса (например, "male" или "female").

    Example:
        >>> get_voice_for_language("ru")
        'male'
        >>> get_voice_for_language("ru", female=True)
        'female'
        >>> get_voice_for_language("xx")
        'male'
    """
    if language is None:
        return default_voice

    lang = language.lower().strip()
    key = f"{lang}_female" if female else lang

    # Пробуем точный ключ, затем базовый язык, затем дефолт
    if key in LANGUAGE_VOICE_MAP:
        return LANGUAGE_VOICE_MAP[key]
    if lang in LANGUAGE_VOICE_MAP:
        return LANGUAGE_VOICE_MAP[lang]
    return default_voice


def supported_languages() -> list:
    """
    Возвращает список поддерживаемых языковых кодов (без _female).

    Returns:
        Список кодов языков, например ["ru", "en", "tr", ...].

    Example:
        >>> langs = supported_languages()
        >>> print(len(langs))
        18
    """
    return sorted([k for k in LANGUAGE_VOICE_MAP if "_female" not in k])
