import pandas as pd
from main import PDF_LINKS_EXCEL, EXCEL_FILE_PATH


def get_bluereport_pdf_dataframe(excel_path):
    dataframe = pd.read_excel(excel_path)
    dataframe.dropna(subset=['URL', 'Tonalität'], inplace=True)
    dataframe['Pdf Titel'] = dataframe['Titel'].str.replace(' ', '_')
    dataframe = dataframe[dataframe['URL'].str.contains('app.bluereport.net')]
    dataframe.drop_duplicates(subset='ID', inplace=True)
    return dataframe


bluereport_pdf_files = get_bluereport_pdf_dataframe(EXCEL_FILE_PATH)
bluereport_pdf_files = bluereport_pdf_files['URL']
bluereport_pdf_files.reset_index(drop=True, inplace=True)

# Create new dataframe with only the URLS
num_col_new_df = len(bluereport_pdf_files) // 50 + 1
num_row_new_df = 50
column_names = [f'URL{i}' for i in range(1, num_col_new_df + 1)]
pdf_links_df = pd.DataFrame(index=range(num_row_new_df), columns=column_names)

num_rows = len(bluereport_pdf_files)
col_counter = 0
final_df = pd.DataFrame()
for i in range(num_col_new_df):
    start_index = i * num_row_new_df
    end_index = (i + 1) * num_row_new_df

    subset = bluereport_pdf_files.iloc[start_index:end_index]
    subset_df = pd.DataFrame(subset)
    subset_df.reset_index(drop=True, inplace=True)
    final_df['URL' + str(i)] = subset_df

final_df.to_excel(PDF_LINKS_EXCEL, index=False)
