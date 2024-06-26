from main import EXCEL_FOLDER
import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_excel("C:\\Users\\tijst\\OneDrive\\Bureaublad\\DataMasterThesis\\Excel\\raw_output.xlsx")
data['sentiment'] = data['sentiment'].map({
    0: "negativ",
    1: "leicht negativ",
    2: "ausgeglichen",
    3: "leicht positiv",
    4: "positiv"}
)

data['sentiment'] = data['sentiment'].replace({'negativ': 0, 'leicht negativ': 1, 'ausgeglichen': 2, 'positiv': 3, 'leicht positiv': 4})
THEME_MAPPINGS = {
    0: "Ländliche Entwicklung, Digitale Innovation",
    1: "Lebensmittelsicherheit, Tiergesundheit",
    2: "Landwirtschaftliche Erzeugung, Gartenbau, Agrarpolitik",
    3: "Gesundheitlicher Verbraucherschutz, Ernährung, Produktsicherheit",
    4: "Sonstiges",
    5: "Wald, Nachhaltigkeit, Nachwachsende Rohstoffe",
    6: "EU-Angelegenheiten, Internationale Zusammenarbeit, Fischerei",
    7: "Agrarmärkte, Ernährungswirtschaft, Export",
    8: "Zentralabteilung"
}

data['aspect'] = data['aspect'].map(THEME_MAPPINGS)

splitted_data_test = data.groupby('aspect').apply(lambda x: train_test_split(x, test_size=0.2, stratify=x['sentiment']))

train_data = pd.concat([item[0] for item in splitted_data_test])
test_data = pd.concat([item[1] for item in splitted_data_test])

splitted_data_train = train_data.groupby('aspect').apply(lambda x: train_test_split(x, test_size=0.2, stratify=x['sentiment']))

train_data = pd.concat([item[0] for item in splitted_data_train])
validation_data = pd.concat([item[1] for item in splitted_data_train])

test_data.to_excel(EXCEL_FOLDER + "\\test_data.xlsx", index=False)
train_data.to_excel(EXCEL_FOLDER + "\\train_data.xlsx", index=False)
validation_data.to_excel(EXCEL_FOLDER + "\\validation_data.xlsx", index=False)

a = 1
