from pdf_title_merge import PDF_PATH_DATAFRAME
import pypdfium2 as pdfium


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


full_text_df = PDF_PATH_DATAFRAME
for index, row in full_text_df.iterrows():
    pdf_path = row['Pdf file Path']  # Replace with the path to your PDF file
    full_text = pdfium_get_text(pdf_path)
    clean_text = clean_full_text(full_text, row['Titel'])
    full_text_df.at[index, 'full text'] = clean_text

a = 1
