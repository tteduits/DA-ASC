import xml.etree.ElementTree as ET
from main import EXCEL_FOLDER
import pandas as pd
from collections import defaultdict
from data_functions import bt_translate_text, ka_perform, eda_perform


def create_csv_data(path, augmentation):
    tree = ET.parse(path)
    root = tree.getroot()

    # Initialize lists to store data
    reviews_data = []

    # Iterate over each Review element
    for review in root.findall('Review'):
        review_id = review.attrib['rid']

        # Initialize review text and dictionaries for categories and polarities
        review_text = ''
        categories = []
        polarities = []

        # Iterate over each sentence within sentences
        sentences = review.find('sentences')
        if sentences is not None:
            for sentence in sentences.findall('sentence'):
                sentence_text = sentence.find('text').text.strip()
                review_text += sentence_text + ' '  # Concatenate sentence text

                # Extract Opinions if present
                opinions = sentence.find('Opinions')
                if opinions is not None:
                    for opinion in opinions.findall('Opinion'):
                        category = opinion.attrib.get('category', 'NULL')
                        polarity = opinion.attrib.get('polarity', 'NULL')
                        categories.append(category)
                        polarities.append(polarity)

        # Append data to list (only if review_text is not empty)
        if review_text:
            reviews_data.append({
                'Review_ID': review_id,
                'Review_Text': review_text.strip(),  # Remove trailing whitespace
                'Categories': categories,
                'Polarities': polarities
            })

    df_reviews = pd.DataFrame(reviews_data)

    if augmentation == 'bt':
        print('start')
        bt_reviews1 = bt_translate_text(df_reviews, 5, 1)
        print('end1')
        bt_reviews2 = bt_translate_text(df_reviews, 4, 2)
        # print('end2')
        bt_reviews3 = bt_translate_text(df_reviews, 3, 1)
        bt_reviews4 = bt_translate_text(df_reviews, 6, 1)
        # print('END3')
        df_reviews = pd.concat([df_reviews, bt_reviews1])
        df_reviews = pd.concat([df_reviews, bt_reviews2])
        df_reviews = pd.concat([df_reviews, bt_reviews3])
        df_reviews = pd.concat([df_reviews, bt_reviews4])

    elif augmentation == 'ka':
        ka_reviews1 = ka_perform(df_reviews)
        ka_reviews2 = ka_perform(df_reviews)
        ka_reviews3 = ka_perform(df_reviews)
        ka_reviews4 = ka_perform(df_reviews)
        df_reviews = pd.concat([df_reviews, ka_reviews1])
        df_reviews = pd.concat([df_reviews, ka_reviews2])
        df_reviews = pd.concat([df_reviews, ka_reviews3])
        df_reviews = pd.concat([df_reviews, ka_reviews4])

    elif augmentation == 'eda':
        eda_reviews1 = eda_perform(df_reviews)
        eda_reviews2 = eda_perform(df_reviews)
        eda_reviews3 = eda_perform(df_reviews)
        eda_reviews4 = eda_perform(df_reviews)
        df_reviews = pd.concat([df_reviews, eda_reviews1])
        df_reviews = pd.concat([df_reviews, eda_reviews2])
        df_reviews = pd.concat([df_reviews, eda_reviews3])
        df_reviews = pd.concat([df_reviews, eda_reviews4])

    # Define a mapping dictionary for polarity
    polarity_map = {
        'negative': 0,
        'neutral': 1,
        'positive': 2
    }

    # Convert Polarities list to numerical values using apply and lambda function
    df_reviews['Polarity_Num'] = df_reviews['Polarities'].apply(lambda x: [polarity_map[item] for item in x])

    # Initialize a defaultdict to store sums and counts for each category in each review
    review_category_polarity_stats = defaultdict(lambda: defaultdict(lambda: {'sum': 0, 'count': 0}))

    # Iterate over each row in df_reviews
    for index, row in df_reviews.iterrows():
        review_id = row['Review_ID']
        categories = row['Categories']
        polarities = row['Polarity_Num']

        # Iterate over each category and polarity in the current review
        for category, polarity in zip(categories, polarities):
            review_category_polarity_stats[review_id][category]['sum'] += polarity
            review_category_polarity_stats[review_id][category]['count'] += 1

    # Calculate average polarity for each category within each review
    average_polarity_dict = {}
    for review_id, category_stats in review_category_polarity_stats.items():
        for category, stats in category_stats.items():
            if stats['count'] > 0:
                polarity_average = stats['sum'] / stats['count']
                if polarity_average < 0.67:
                    average_polarity_dict[(review_id, category)] = 0
                elif 0.67 <= polarity_average < 1.33:
                    average_polarity_dict[(review_id, category)] = 1
                else:
                    average_polarity_dict[(review_id, category)] = 2
            else:
                average_polarity_dict[
                    (review_id, category)] = None  # Handle case where there are no reviews for a category

    # Create a new column with the average polarity dictionary
    df_reviews['Average_Polarity'] = df_reviews.apply(
        lambda row: {category: average_polarity_dict.get((row['Review_ID'], category)) for category in
                     row['Categories']},
        axis=1)

    reshaped_data = []
    for idx, row in df_reviews.iterrows():
        for aspect, sentiment in row['Average_Polarity'].items():
            reshaped_data.append({
                'Review_ID': row['Review_ID'],
                'aspect': aspect,
                'Review_Text': row['Review_Text'],
                'sentiment': sentiment
            })

    data = pd.DataFrame(reshaped_data)

    data.to_csv(EXCEL_FOLDER + '\\SemEval_Laptop_' + augmentation + 'train4.csv', index=False)
    a = 1


train_data_path = EXCEL_FOLDER + '\\ABSA16_Laptops_Train_SB1_v2.xml'
test_data_path = EXCEL_FOLDER + '\\EN_LAPT_SB1_TEST_Gold.xml'

create_csv_data(test_data_path, '')
# create_csv_data(train_data_path, 'bt')
# create_csv_data(train_data_path, 'ka')
# create_csv_data(train_data_path, 'eda')
