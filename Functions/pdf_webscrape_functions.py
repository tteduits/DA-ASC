import pypdfium2 as pdfium
import requests
from bs4 import BeautifulSoup


def clean_full_text(text, titel):
    index_of_first = text.find(titel)

    # Find the index of the second occurrence of the word
    index_of_second = text.find(titel, index_of_first + len(titel))

    # If both occurrences are found and the second occurrence is after the first one
    if index_of_first != -1 and index_of_second != -1 and index_of_second > index_of_first:
        # Remove the first occurrence and include the second occurrence and everything after it
        text = text[:index_of_first] + text[index_of_second:]
    if index_of_first != -1:
        text = text[index_of_first:]

    text = text.replace("nur zum internen gebrauch ", "")
    text = text.lower()

    return text


def pdfium_get_text(data: bytes) -> str:
    text = ""
    pdf = pdfium.PdfDocument(data)
    for i in range(len(pdf)):
        page = pdf.get_page(i)
        textpage = page.get_textpage()
        text += textpage.get_text_range() + "\n"
    text = text.lower()

    return text


def do_web_scrape(url, quelle, titel):
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    if quelle == 'merkur.de':
        paragraphs = soup.find_all(['h3', 'h2', 'p', 'li'], class_=['id-StoryElement-factBox-headline', 'id-StoryElement-factBox-paragraph', 'id-StoryElement-paragraph', 'id-StoryElement-crosshead', 'id-StoryElement-list-item', 'id-StoryElement-leadText'])
        full_text = titel + ' '
    else:
        return None

    for paragraph in paragraphs:
        full_text = full_text + paragraph.get_text(strip=True) + " "

    return full_text
