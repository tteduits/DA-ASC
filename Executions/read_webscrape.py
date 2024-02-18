from main import WEB_SCRAPE_EXCEL
import pandas as pd
from Executions.read_pdf import clean_full_text
from Functions.pdf_webscrape_functions import do_web_scrape


web_scrape_df = pd.read_excel(WEB_SCRAPE_EXCEL)
quelle_unique_values = web_scrape_df['Quelle'].unique()

for index, row in web_scrape_df.iterrows():
    full_text = do_web_scrape(row['URL'], row['Quelle'], row['Titel'])
    if full_text is None:
        continue
    cleaned_text = clean_full_text(full_text, row['Titel'])
    a = 1
