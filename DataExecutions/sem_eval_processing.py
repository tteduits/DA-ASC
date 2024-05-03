from main import SEM_EVAL_PATH
import xml.etree.ElementTree as ET
import pandas as pd


def retrieve_xml_sem_eval(path):
    tree = ET.parse(path)
    root = tree.getroot()

    data = {'Review_ID': [],
            'Sentence': [],
            'Opinion_Target': [],
            'Category': [],
            'Polarity': []}
    count = 0
    for review in root.findall('Review'):
        count += 1
        review_id = review.attrib['rid']
        for sentence in review.find('sentences'):
            sentence_text = sentence.find('text').text
            opinions = sentence.find('Opinions')
            if opinions is not None:
                for opinion in opinions.findall('Opinion'):
                    target = opinion.attrib['target']
                    category = opinion.attrib['category']
                    polarity = opinion.attrib['polarity']

                    data['Review_ID'].append(review_id)
                    data['Sentence'].append(sentence_text)
                    data['Opinion_Target'].append(target)
                    data['Category'].append(category)
                    data['Polarity'].append(polarity)

    df = pd.DataFrame(data)
    return df


train_df = retrieve_xml_sem_eval(SEM_EVAL_PATH+"\\ABSA-15_Restaurants_Train_Final.xml")
val_df = retrieve_xml_sem_eval(SEM_EVAL_PATH+"\\ABSA15_Restaurants_Test.xml")
combined_df = pd.concat([train_df, val_df], ignore_index=True)
cross_tab = pd.crosstab(combined_df['Category'], combined_df['Polarity'])
cross_tab.loc['Cumulative aspect'] = cross_tab.sum(axis=0)
cross_tab['Cumulative sentiment'] = cross_tab.sum(axis=1)


a = 1
