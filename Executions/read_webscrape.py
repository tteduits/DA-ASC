from main import WEB_SCRAPE_EXCEL
import pandas as pd
from Functions.pdf_webscrape_functions import do_web_scrape, clean_full_text


web_scrape_df = pd.read_excel(WEB_SCRAPE_EXCEL)
quelle_unique_values = web_scrape_df['Quelle'].unique()
web_scrape_df['full text'] = pd.Series(dtype=float)

for index, row in web_scrape_df.iterrows():
    full_text = do_web_scrape(row['URL'], row['Quelle'], row['Titel'])
    if full_text is None:
        continue
    cleaned_text = clean_full_text(full_text, row['Titel'])
    web_scrape_df.at[index, 'full text'] = cleaned_text

web_scrape_df.to_excel('D:\\MasterThesisTijs\\Excel\\excel_full_text_webscrape.xlsx', index=False)
