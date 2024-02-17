from pdf_title_merge import PDF_PATH_DATAFRAME
import pdfplumber
import re


def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    text = text.lower()

    return text


def clean_full_text(text, titel):
    index_of_first = text.find(titel)

    # Find the index of the second occurrence of the word
    index_of_second = text.find(titel, index_of_first + len(titel))

    # If both occurrences are found and the second occurrence is after the first one
    if index_of_first != -1 and index_of_second != -1 and index_of_second > index_of_first:
        # Remove the first occurrence and include the second occurrence and everything after it
        text = text[:index_of_first] + text[index_of_second:]

    index_of_as = text.find(titel)
    text = text[index_of_as:]
    text = text.replace("nur zum internen gebrauch ", "")

    return text


for index, row in PDF_PATH_DATAFRAME.iterrows():
    pdf_path = row['Pdf file Path']  # Replace with the path to your PDF file
    full_text = extract_text_from_pdf(pdf_path)
    clean_text = clean_full_text(full_text, row['Titel'])
    # print(full_text)
