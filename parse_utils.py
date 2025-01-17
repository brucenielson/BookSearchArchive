import re
from typing import Dict, Set
import csv


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


# TODO: Change code to use this version
def load_sections_to_skip(csv_path) -> Dict[str, Set[str]]:
    sections_to_skip: Dict[str, Set[str]] = {}
    # if self._is_directory:
    #     csv_path = Path(self._file_paths[0]) / self._skip_file
    # else:
    #     # Get the directory of the file and then look for the csv file in that directory
    #     csv_path = Path(self._file_paths[0]).parent / self._skip_file

    if csv_path.exists():
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader: csv.DictReader[str] = csv.DictReader(csvfile)
            row: dict[str, str]
            for row in reader:
                book_title: str = row['Book Title'].strip()
                section_title: str = row['Section Title'].strip()
                if book_title and section_title:
                    if book_title not in sections_to_skip:
                        sections_to_skip[book_title] = set()
                    sections_to_skip[book_title].add(section_title)

        # Count total sections to skip across all books
        skip_count: int = sum(len(sections) for _, sections in sections_to_skip.items())
        print(f"Loaded {skip_count} sections to skip.")
    else:
        print("No sections_to_skip.csv file found. Processing all sections.")

    return sections_to_skip
