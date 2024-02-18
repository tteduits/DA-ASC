import pandas as pd


def get_bluereport_pdf_dataframe(excel_path):
    dataframe = pd.read_excel(excel_path)
    dataframe.dropna(subset=['URL', 'Tonalität'], inplace=True)
    dataframe['Pdf Titel'] = dataframe['Titel'].str.replace(' ', '_')
    dataframe = dataframe[dataframe['URL'].str.contains('app.bluereport.net')]
    dataframe.drop_duplicates(subset='ID', inplace=True)
    return dataframe


def create_web_scraping_df(excel_path, minimal_occurences):
    dataframe = pd.read_excel(excel_path)
    dataframe.dropna(subset=['URL', 'Tonalität'], inplace=True)
    dataframe = dataframe[~dataframe['URL'].str.contains('app.bluereport.net') & ~dataframe['Mediengattung'].str.contains('tv')]
    dataframe = dataframe.sort_values(by='Quelle')
    quelle_counts = dataframe['Quelle'].value_counts()
    ordered_quelle = quelle_counts.index.tolist()
    dataframe['Quelle'] = pd.Categorical(dataframe['Quelle'], categories=ordered_quelle, ordered=True)
    dataframe = dataframe.sort_values(by='Quelle')

    # Get the values whose counts are greater than or equal to 'x'
    valid_values = quelle_counts[quelle_counts >= minimal_occurences].index.tolist()
    dataframe = dataframe[dataframe['Quelle'].isin(valid_values)]

    return dataframe
