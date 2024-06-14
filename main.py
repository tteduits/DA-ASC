EXTERNAL_DISK = 'D:\\MasterThesisTijs\\PdfFiles'
DATA_FOLDER = 'C:\\Users\\tijst\\OneDrive\\Bureaublad\\DataMasterThesis\\'
EXCEL_FOLDER = 'D:\\MasterThesisTijs\\Excel'
SEM_EVAL_PATH = 'C:\\Users\\tijst\\OneDrive\\Bureaublad\\DataMasterThesis\\SemEval\\'
PDF_LINKS_EXCEL = 'C:\\Users\\tijst\\OneDrive\\Bureaublad\\DataMasterThesis\\Excel\\PDF_files\\'
FULL_TEXT_FOLDER = 'C:\\Users\\tijst\\OneDrive\\Bureaublad\\DataMasterThesis\\Excel\\FullText\\'
PDF_FULL_TEXT_PATH = 'C:\\Users\\tijst\\OneDrive\\Bureaublad\\DataMasterThesis\\Excel\\excel_full_text_pdf.xlsx'

COL_LENGTH = 10
ROW_LENGTH = 34

SENTIMENT_MAPPINGS = {
    "negativ": 0,
    "leicht negativ": 1,
    "ausgeglichen": 2,
    "leicht positiv": 3,
    "positiv": 4
}

THEME_MAPPINGS = {
    "Ländliche Entwicklung, Digitale Innovation": 0,
    "Lebensmittelsicherheit, Tiergesundheit": 1,
    "Landwirtschaftliche Erzeugung, Gartenbau, Agrarpolitik": 2,
    "Gesundheitlicher Verbraucherschutz, Ernährung, Produktsicherheit": 3,
    "Sonstiges": 4,
    "Wald, Nachhaltigkeit, Nachwachsende Rohstoffe": 5,
    "EU-Angelegenheiten, Internationale Zusammenarbeit, Fischerei": 6,
    "Agrarmärkte, Ernährungswirtschaft, Export": 7,
    "Zentralabteilung": 8,
}

FASTTEXT_PARAM = {
    'input': DATA_FOLDER + 'fasttext_train_data.txt',
    'model': 'skipgram',

    'lr': 0.05,
    'wordNgrams': 5,
    'dim': 300,
    't': 1e-4,
    'neg': 5,
    'loss': 'ns',
    'epoch': 10

}
