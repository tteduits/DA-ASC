import pandas as pd
from main import EXCEL_FOLDER
import stanza

train_data = pd.read_excel(EXCEL_FOLDER + '\\NewsArticle\\validation_data.xlsx')
test_data = pd.read_excel(EXCEL_FOLDER + '\\NewsArticle\\test_data.xlsx')

nlp = stanza.Pipeline(lang='de', processors='tokenize,ner')
terms_of_interest = {'Bundesministerium für Ernährung und Landwirtschaft', 'undeslandwirtschaftsministerium',
                     'Bundesernährungsministerium', 'Bundesagrarministerium', 'Bundesfischereiministerium',
                     'Bundeswaldministerium', 'Bundesforstministerium', 'özdemir',
                     'Bundesminister für Ernährung und Landwirtschaft', 'Bundeslandwirtschaftsminister',
                     'Bundesernährungsminister', 'Bundesagrarminister', 'Bundesfischereiminister', 'Bundeswaldminister',
                     'Bundesforstminister', 'Rottmann', 'Nick', 'Bender', 'Müller'}
terms_of_interest = {term.lower() for term in terms_of_interest}

for idx, row in train_data.iterrows():
    text = row['clean_text']

    doc = nlp(text)
    filtered_sentences = []

    for sentence in doc.sentences:
        if any(term in sentence.text.lower() for term in terms_of_interest):
            filtered_sentences.append(sentence.text)

    filtered_text = ' '.join(filtered_sentences)
    train_data['ner_text'] = filtered_text

train_data.to_excel(EXCEL_FOLDER + '\\random_train.xlsx', index=False)
for idx, row in test_data.iterrows():
    text = row['clean_text']

    doc = nlp(text)
    filtered_sentences = []

    for sentence in doc.sentences:
        if any(term in sentence.text.lower() for term in terms_of_interest):
            filtered_sentences.append(sentence.text)

    filtered_text = ' '.join(filtered_sentences)
    test_data['ner_text'] = filtered_text

test_data.to_excel(EXCEL_FOLDER + '\\random_test.xlsx', index=False)
a = 1
