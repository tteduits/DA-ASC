import pandas as pd


def get_bluereport_pdf_dataframe(excel_path):
    dataframe = pd.read_excel(excel_path)
    dataframe.dropna(subset=['URL', 'Tonalität'], inplace=True)
    dataframe['Pdf Titel'] = dataframe['Titel'].str.replace(' ', '_')
    dataframe = dataframe[dataframe['URL'].str.contains('app.bluereport.net')]
    dataframe.drop_duplicates(subset='ID', inplace=True)
    return dataframe
