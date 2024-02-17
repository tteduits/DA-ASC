import glob
import os
from Levenshtein import ratio
from create_excel_pdf import get_bluereport_pdf_dataframe
from main import EXCEL_FILE_PATH, PDF_ARTICLE_FOLDER
import pandas as pd
import re
from fuzzywuzzy import fuzz


bluereport_pdf_dataframe = get_bluereport_pdf_dataframe(EXCEL_FILE_PATH)
pdf_article_names_path = glob.glob(PDF_ARTICLE_FOLDER + '/*.pdf')
pdf_article_names = [os.path.basename(file)[:-4].replace('_', ' ') for file in pdf_article_names_path]
special_chars_pattern = re.compile(r'[^a-zA-Z0-9\säöüÄÖÜß]')

# First do a hard check for pdf name and titel
for index, row in bluereport_pdf_dataframe.iterrows():
    pdf_titel = row['Pdf Titel']

    # Remove special characters (except those specific to German language) and convert to lowercase
    cleaned_pdf_titel = special_chars_pattern.sub('', pdf_titel).lower()

    # Iterate over matching filenames
    matching_files = [file for file in pdf_article_names_path
                      if cleaned_pdf_titel in special_chars_pattern.sub('', file).lower()]

    # If there's a match, set 'Pdf file Path' to the first matching file
    if matching_files:
        bluereport_pdf_dataframe.at[index, 'Pdf file Path'] = matching_files[0]

known_pdf_path = bluereport_pdf_dataframe[~bluereport_pdf_dataframe['Pdf file Path'].isna()]
unknown_pdf_path = bluereport_pdf_dataframe[bluereport_pdf_dataframe['Pdf file Path'].isna()]


# Iterate over unknown_pdf_path
for index, row in unknown_pdf_path.iterrows():
    text = row['Pdf Titel'].lower()
    pdf_titel = re.sub(r'[^a-zA-Z0-9äöüß]', '', text)

    # Initialize variables to store the maximum similarity and the most similar filename
    max_similarity = 0
    most_similar_filename = None

    # Iterate over matching filenames
    for file in pdf_article_names_path:
        text = file.lower()
        file_processed = re.sub(r'[^a-zA-Z0-9äöüß]', '', text)

        # Calculate similarity between Pdf Titel and filename
        similarity = fuzz.ratio(pdf_titel, file_processed)

        # Update max_similarity and most_similar_filename if similarity is higher
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_filename = file

    # Set Pdf file Path and similarity score
    unknown_pdf_path.at[index, 'Pdf file Path'] = most_similar_filename
    unknown_pdf_path.at[index, 'Similarity Pdf file name and titel'] = max_similarity


df_sorted = unknown_pdf_path.sort_values(by='Similarity Pdf file name and titel')


known_pdf_path = bluereport_pdf_dataframe[~bluereport_pdf_dataframe['Pdf file Path'].isna()]
unknown_pdf_path = bluereport_pdf_dataframe[bluereport_pdf_dataframe['Pdf file Path'].isna()]

a =1
