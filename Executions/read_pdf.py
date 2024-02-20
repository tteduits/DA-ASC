from merge_title_pdf import PDF_PATH_DATAFRAME
from Functions.pdf_read_functions import pdfium_get_text, clean_full_text


full_text_df = PDF_PATH_DATAFRAME
for index, row in full_text_df.iterrows():
    pdf_path = row['Pdf file Path']  # Replace with the path to your PDF file
    full_text = pdfium_get_text(pdf_path)
    clean_text = clean_full_text(full_text, row['Titel'])
    full_text_df.at[index, 'full text'] = clean_text

full_text_df.to_excel('D:\\MasterThesisTijs\\Excel\\excel_full_text_pdf.xlsx')

