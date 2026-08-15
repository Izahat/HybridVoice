"""
Маппинг языков на голоса (Voice Design).

Для каждого языка по умолчанию используется мужской голос.
Ключ с суффиксом "_female" даёт женский голос.

Значения — это instruct-строки для OmniVoice Voice Design режима.
"""

from typing import Optional


# ============================================================
# Маппинг язык → instruct-голос (по умолчанию = мужской)
# ============================================================
# Ключ "xx"        → мужской голос для языка xx
# Ключ "xx_female" → женский голос для языка xx
# OmniVoice поддерживает 646 языков! Этот маппинг — только
# для популярных. Для остальных просто передай код в language=.
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

    Для языка "ru" вернёт "male", для "ru" + female=True вернёт "female".
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
    Возвращает список кодов языков с настроенным маппингом голоса.

    ВНИМАНИЕ: OmniVoice поддерживает 646 языков! Этот список содержит
    только языки с настроенным маппингом (male/female). Для любого
    другого языка передай код напрямую в language= — OmniVoice
    сгенерирует речь с голосом по умолчанию.

    Returns:
        Список кодов языков с маппингом.

    Example:
        >>> langs = supported_languages()
        >>> print(len(langs))
        18
    """
    return sorted([k for k in LANGUAGE_VOICE_MAP if "_female" not in k])


def all_languages() -> dict:
    """
    Возвращает СЛОВАРЬ ВСЕХ 646 языков, которые поддерживает OmniVoice.

    Ключ — код языка ISO 639, значение — название на английском.

    Returns:
        Словарь {код: название}, например {"ru": "russian", "en": "english", ...}.

    Example:
        >>> langs = all_languages()
        >>> print(len(langs))
        646
        >>> print(langs.get("ru"))
        russian
        >>> print(langs.get("sw"))
        swahili
    """
    try:
        from omnivoice.utils.lang_map import LANG_IDS, LANG_NAME_TO_ID

        code_to_name = {v: k for k, v in LANG_NAME_TO_ID.items()}
        return {code: code_to_name.get(code, code) for code in sorted(LANG_IDS)}
    except ImportError:
        return {}


def list_languages(query: Optional[str] = None) -> None:
    """
    Печатает список всех языков OmniVoice с кодами и названиями.

    Без аргументов — выводит все 646 языков.
    С аргументом — фильтрует по коду или названию.

    Args:
        query: Фильтр (строка). Ищет совпадение в коде или названии.
            Если None — выводит все языки.

    Example:
        >>> list_languages()            # все 646 языков
        >>> list_languages("ru")        # языки с "ru" в коде
        >>> list_languages("arabic")    # все арабские языки
        >>> list_languages("kazakh")    # казахский
    """
    langs = all_languages()
    if not langs:
        print("OmniVoice не установлен. Установите: pip install omnivoice")
        return

    if query:
        q = query.lower()
        filtered = {
            code: name
            for code, name in langs.items()
            if q in code.lower() or q in name.lower()
        }
        if not filtered:
            print(f"Языки по запросу '{query}' не найдены.")
            return
        langs = filtered

    print(f"{'КОД':6} {'НАЗВАНИЕ'}")
    print("-" * 50)
    for code, name in sorted(langs.items()):
        print(f"{code:6} {name}")
    print("-" * 50)
    print(f"Найдено: {len(langs)} языков")
