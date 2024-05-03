import pandas as pd
from main import EXCEL_FOLDER
from sklearn.model_selection import train_test_split
import random


combined_data = pd.read_excel(EXCEL_FOLDER + "\\combined_data.xlsx")
combined_data = combined_data.drop(combined_data.columns[0], axis=1)
combined_data['aspect'] = combined_data['aspect'].apply(eval)
df_single_value = combined_data[combined_data['aspect'].apply(lambda x: len(x) == 1)]
df_multiple_values = combined_data[combined_data['aspect'].apply(lambda x: len(x) > 1)]

unique_aspects = df_single_value['Thema'].unique()

train_df = pd.DataFrame()
test_df = pd.DataFrame()
for topic in unique_aspects:
    topic_df = combined_data[combined_data['Thema'] == topic].copy()

    train_topic_df, test_topic_df = train_test_split(topic_df, train_size=0.8, stratify=topic_df['sentiment'], random_state=42)
    train_df = pd.concat([train_df, train_topic_df], ignore_index=True)
    test_df = pd.concat([test_df, test_topic_df], ignore_index=True)

df_part1 = df_multiple_values.sample(frac=0.5, random_state=42)
df_part2 = df_multiple_values.drop(df_part1.index)

df_part1.reset_index(drop=True, inplace=True)
df_part2.reset_index(drop=True, inplace=True)

train_df = pd.concat([train_df, df_part2], ignore_index=True)
test_df = pd.concat([test_df, df_part1], ignore_index=True)
a = 1
train_df.to_excel(EXCEL_FOLDER + '\\train_data.xlsx', index=False)
test_df.to_excel(EXCEL_FOLDER + '\\test_data.xlsx', index=False)

