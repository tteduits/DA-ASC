import pandas as pd
import math
from main import PDF_LINKS_EXCEL, EXCEL_FOLDER, SENTIMENT_MAPPINGS, THEME_MAPPINGS, COL_LENGTH, ROW_LENGTH
from DataFunctions.clean_transform_data import remove_duplicate_themes


data = pd.read_excel('C:\\Users\\tijst\\Downloads\\BMEL Medienresonanzanalyse Masterdatenblatt (Q1 2021-Q1 2024).xlsx')
data = data[data['URL'].notna() & data['URL'].str.contains('bluereport')]

data['aspect'] = data['Thema'].map(THEME_MAPPINGS)
data['sentiment'] = data['Tonalität'].map(SENTIMENT_MAPPINGS)
Thema = data['Thema'].unique()
Tonalität = data['Tonalität'].unique()
matrix_data = data.groupby(['Thema', 'Tonalität']).size().unstack(fill_value=0)
sorted_values = sorted(Thema)
matrix_data.loc['Cumulative Thema'] = matrix_data.sum(axis=0)
matrix_data['Cumulative Tonalität'] = matrix_data.sum(axis=1)
print(matrix_data.index)

data_combined = data.groupby(['Titel', 'Quelle', 'Mediengattung', 'Veröffentlichungsdatum']).agg({
    'ID': 'first',
    'URL': 'first',
    'Tonalität': lambda x: x.tolist(),
    'Thema': lambda x: x.tolist(),
    'sentiment': lambda x: x.tolist(),
    'aspect': lambda x: x.tolist()
}).reset_index()

data_combined = remove_duplicate_themes(data_combined)
data_combined['Titel'] = data_combined['Titel'].str.lower()
data_combined_sorted = data_combined.sort_values(by='ID')
data_combined.to_excel(EXCEL_FOLDER+"combined_data.xlsx", index=False)
num_rows = len(data_combined)
num_csv_needed = math.ceil(num_rows / (COL_LENGTH*ROW_LENGTH))
csv_counter = 0

for csv_number in range(num_csv_needed+1):
    pdf_links_df = pd.DataFrame()

    subset_URL = data_combined.iloc[COL_LENGTH*ROW_LENGTH*csv_number:COL_LENGTH*ROW_LENGTH*(csv_number+1)]['URL']
    subset_df = data_combined.iloc[COL_LENGTH*ROW_LENGTH*csv_number:COL_LENGTH*ROW_LENGTH*(csv_number+1)]
    for col in range(COL_LENGTH):
        url_col = ((subset_URL.iloc[ROW_LENGTH*col:ROW_LENGTH*(col+1)]).reset_index())['URL']
        pdf_links_df['URL' + str(col)] = pd.Series(url_col)

    pdf_links_df.to_excel(PDF_LINKS_EXCEL + 'pdf_file' + str(csv_number) + '.xlsx', index=False)
    pdf_links_df.to_excel(PDF_LINKS_EXCEL + 'pdf_file_' + str(csv_number) + '.xlsx', index=False)
    subset_df.to_excel(EXCEL_FOLDER+'CombinedData\\combined_data' + str(csv_number) + '.xlsx', index=False)
    csv_counter += 1
