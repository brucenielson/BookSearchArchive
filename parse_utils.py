import re


def is_roman_numeral(s: str) -> bool:
    roman_numeral_pattern = r'(?i)^(M{0,3})(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
    return bool(re.match(roman_numeral_pattern, s.strip()))


def enhance_title(text: str) -> str:
    text = text.strip()
    # If all caps but not a roman numeral and not first word before a space of a sentence roman numeral
    if text.isupper() and not is_roman_numeral(text):
        # If first word before a space is a roman numeral, leave that part as is
        first_word = text.split(' ', 1)[0]
        if is_roman_numeral(first_word) and first_word != text:
            text = first_word + text[len(first_word):].title()
        else:
            # If all caps, title case
            text = text.title()
        # Replace 'S with 's after title casing
        text = text.replace("'S", "'s")
        text = text.replace("’S", "’s")
    return text
