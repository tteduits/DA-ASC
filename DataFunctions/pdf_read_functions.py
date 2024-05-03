import fitz
import difflib


def clean_full_text(text, title):
    text = text.lower()
    text = text.replace('-\n', ' ')
    text = text.replace('\n', ' ')
    text = text.replace("nur zum internen gebrauch", "")
    text = text.replace("  ", " ")
    text = add_spaces_after_period(text)
    title_found = False

    first_index = text.find(title)
    if first_index != -1:
        second_index = text.find(title, first_index + len(title))
        if second_index != -1:
            text = text[second_index:]
        else:
            text = text[first_index:]

    if title_found:
        return text

    for i in range(len(text) - len(title)):
        similarity = difflib.SequenceMatcher(None, text[i:i+len(title)], title).ratio()
        if similarity >= 0.75:
            text = text[i:]
            break

    return text


def read_pdf(data: bytes) -> str:
    text = ""
    pdf_document = fitz.open(data)
    for page in pdf_document:
        text += page.get_text()

    pdf_document.close()

    return text


def add_spaces_after_period(text):
    """
    Add spaces after periods where needed in the text.
    """
    corrected_text = ""
    for i, char in enumerate(text):
        if char == ".":
            if i + 1 < len(text) and (i == len(text) - 1 or text[i + 1] != " ") and (i + 1 == len(text) or not text[i + 1].isdigit()):
                corrected_text += char + " "
            else:
                corrected_text += char
        else:
            corrected_text += char
    return corrected_text


def process_row_reading(row):
    full_text = read_pdf(row['path'])
    clean_text = clean_full_text(full_text, row['Titel'])
    return clean_text, full_text
