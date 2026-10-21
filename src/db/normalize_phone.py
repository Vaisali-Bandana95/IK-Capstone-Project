import re


def normalize_phone(phone: str | None) -> str:
    if phone is None:
        return ""

    has_international_prefix = phone.lstrip().startswith("+")
    digits = re.sub(r"\D", "", phone)
    return f"+{digits}" if has_international_prefix else digits
