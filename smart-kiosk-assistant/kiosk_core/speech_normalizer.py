"""Speech normalization: what the kiosk SAYS, separate from what it displays.

The transcript/UI keep "₹169" and "8 AM". The TTS engine gets "one hundred
sixty nine rupees" and "eight AM", because speecht5 reads "₹169" as
literal symbol-and-digit tokens rather than a spoken price.

Deterministic string work, microseconds per phrase, no model involved.
Applied inside ``TtsClient.synthesize_to_file`` so every call path (streamed
clauses, the pre-synthesized opener) is covered and nothing can bypass it.

Adapted from the reference prototype's ``pipeline/speech.py``
(kiosk-voice-lab-main), which is USD/``$``-only. This version adds ``₹``
(and ``Rs``/``INR``) support for our restaurant's rupee prices; the
underlying number-to-words and time/percent/unit handling is unchanged.
"""
import re

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve "
         "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety")

_UNITS = {"oz": ("ounce", "ounces"), "lb": ("pound", "pounds"),
          "ct": ("count", "count"), "pc": ("piece", "piece"),
          "ml": ("milliliter", "milliliters"), "g": ("gram", "grams"),
          "kg": ("kilogram", "kilograms")}


def _under_thousand(n):
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + (" " + _ONES[ones] if ones else "")
    hundreds, rest = divmod(n, 100)
    return _ONES[hundreds] + " hundred" + (" " + _under_thousand(rest) if rest else "")


def number_to_words(n):
    """Integer to spoken English. Handles 0 through the millions."""
    n = int(n)
    if n < 0:
        return "minus " + number_to_words(-n)
    if n < 1000:
        return _under_thousand(n)
    if n < 1_000_000:
        thousands, rest = divmod(n, 1000)
        return (_under_thousand(thousands) + " thousand"
                + (" " + _under_thousand(rest) if rest else ""))
    millions, rest = divmod(n, 1_000_000)
    return (_under_thousand(millions) + " million"
            + (" " + number_to_words(rest) if rest else ""))


def _say_rupees(whole, paise):
    """Indian price idiom: ₹169 is 'one hundred sixty nine rupees', ₹169.50
    is 'one hundred sixty nine rupees and fifty paise'."""
    if paise is None or paise == 0:
        return number_to_words(whole) + (" rupee" if whole == 1 else " rupees")
    rupee_part = number_to_words(whole) + (" rupee" if whole == 1 else " rupees")
    paise_part = number_to_words(paise) + (" paisa" if paise == 1 else " paise")
    return rupee_part + " and " + paise_part


def _say_money(whole, cents):
    """US price idiom: $29.99 is 'twenty nine ninety nine', not a sum of parts."""
    if cents is None or cents == 0:
        return number_to_words(whole) + (" dollar" if whole == 1 else " dollars")
    if whole == 0:
        return number_to_words(cents) + (" cent" if cents == 1 else " cents")
    if whole < 1000:
        spoken_cents = ("oh " + _ONES[cents]) if cents < 10 else number_to_words(cents)
        return number_to_words(whole) + " " + spoken_cents
    return (number_to_words(whole) + " dollars and " + number_to_words(cents)
            + (" cent" if cents == 1 else " cents"))


def _rupee_sub(m):
    whole = int(m.group(1).replace(",", ""))
    paise = int(m.group(2)) if m.group(2) else None
    return _say_rupees(whole, paise)


def _money_sub(m):
    whole = int(m.group(1).replace(",", ""))
    cents = int(m.group(2)) if m.group(2) else None
    return _say_money(whole, cents)


def _time_sub(m):
    hour, minute = int(m.group(1)), int(m.group(2))
    meridiem = (m.group(3) or "").strip()
    spoken = number_to_words(hour)
    if minute:
        spoken += " " + (("oh " + _ONES[minute]) if minute < 10
                         else number_to_words(minute))
    if meridiem:
        spoken += " " + meridiem.upper().replace(".", "")
    return spoken


def _unit_sub(m):
    value, unit = m.group(1), m.group(2).lower()
    singular, plural = _UNITS[unit]
    n = float(value) if "." in value else int(value)
    words = (number_to_words(int(n)) if float(n).is_integer()
             else " point ".join([number_to_words(int(str(n).split(".")[0])),
                                  " ".join(_ONES[int(d)] for d in str(n).split(".")[1])]))
    return words + " " + (singular if n == 1 else plural)


def for_speech(text):
    """Rewrite a phrase the way a person would say it out loud."""
    if not text:
        return text
    t = text
    # numeric ranges written with a dash: "8 AM-11 PM"
    # a time before the dash keeps its AM/PM ("8 AM-11 PM")
    t = re.sub(r"(\d(?:\s*[AaPp][Mm])?)\s*[-–—]\s*([₹$]?\d)", r"\1 to \2", t)
    t = re.sub(r"</?[A-Za-z][A-Za-z0-9_-]{0,30}>", "", t)               # stray tags
    t = re.sub(r"\s*\n\s*[-*\u2022]\s*", ", ", t)                     # bulleted lines read as a list
    t = re.sub(r":\s*,\s*", ": ", t)
    t = re.sub(r"\s*\n+\s*", " ", t)
    # rupees, before any bare-number handling can touch the digits
    t = re.sub(r"₹\s?(\d+)\.\s+(\d{2})\b", r"₹\1.\2", t)                # "₹5. 49" as the model sometimes writes it
    t = re.sub(r"(?:₹|Rs\.?\s?|INR\s?)(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{2}))?\b", _rupee_sub, t)
    # money, before any bare-number handling can touch the digits
    t = re.sub(r"\$\s?(\d+)\.\s+(\d{2})\b", r"$\1.\2", t)               # "$5. 49" as the model sometimes writes it
    t = re.sub(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{2}))?\b", _money_sub, t)
    # clock times
    t = re.sub(r"\b(\d{1,2}):(\d{2})\s*([AaPp]\.?[Mm]\.?)?", _time_sub, t)
    # percentages
    t = re.sub(r"\b(\d+)\s*%", lambda m: number_to_words(m.group(1)) + " percent", t)
    # compressed units: 84ct, 1lb, 4.6oz, 330 ml
    t = re.sub(r"\b(\d+(?:\.\d+)?)\s*(" + "|".join(_UNITS) + r")\b", _unit_sub, t)
    # remaining bare integers, including decimals read as "point"
    t = re.sub(r"\b(\d+)\.(\d+)\b",
               lambda m: number_to_words(m.group(1)) + " point "
                         + " ".join(_ONES[int(d)] for d in m.group(2)), t)
    t = re.sub(r"\b\d+\b", lambda m: number_to_words(m.group(0)), t)
    # tidy artifacts that make speech engines stumble
    t = re.sub(r"\s*[—–]\s*", ", ", t)
    t = t.replace("&", " and ").replace("#", " number ")
    t = re.sub(r"\s+", " ", t).strip()
    return t
