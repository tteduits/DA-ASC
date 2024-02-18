import pandas as pd
from main import PDF_LINKS_EXCEL, EXCEL_FILE_PATH_RAW, WEB_SCRAPE_EXCEL, MINIMAL_OCCURENCES
from Functions.excel_pdf_web_functions import get_bluereport_pdf_dataframe, create_web_scraping_df


bluereport_pdf_files = get_bluereport_pdf_dataframe(EXCEL_FILE_PATH_RAW)
bluereport_pdf_files = bluereport_pdf_files['URL']
bluereport_pdf_files.reset_index(drop=True, inplace=True)

web_scrape_df = create_web_scraping_df(EXCEL_FILE_PATH_RAW, MINIMAL_OCCURENCES)

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
web_scrape_df.to_excel(WEB_SCRAPE_EXCEL, index=False)