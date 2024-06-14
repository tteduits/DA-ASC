import pandas as pd
from main import EXCEL_FOLDER
from sklearn.model_selection import train_test_split
import ast


combined_data = pd.read_excel(EXCEL_FOLDER + "\\combined_data.xlsx")
combined_data = combined_data.drop(combined_data.columns[0], axis=1)
combined_data['aspect'] = combined_data['aspect'].apply(eval)
df_single_value = combined_data[combined_data['aspect'].apply(lambda x: len(x) == 1)]
df_multiple_values = combined_data[combined_data['aspect'].apply(lambda x: len(x) > 1)]

df_single_value['aspect'] = df_single_value['aspect'].apply(lambda x: x[0])
df_single_value['sentiment'] = df_single_value['sentiment'].apply(ast.literal_eval)
df_single_value['sentiment'] = df_single_value['sentiment'].apply(lambda x: x[0])

df_multiple_values['sentiment'] = df_multiple_values['sentiment'].apply(ast.literal_eval)

d1 = df_multiple_values.copy()
d2 = df_multiple_values.copy()

d1['aspect'] = d1['aspect'].apply(lambda x: x[0])
d1['sentiment'] = d1['sentiment'].apply(lambda x: x[0])

d2['aspect'] = d2['aspect'].apply(lambda x: x[1])
d2['sentiment'] = d2['sentiment'].apply(lambda x: x[1])

selected_columns = ['aspect', 'sentiment', 'clean_text']

df_single_value = df_single_value[selected_columns].reset_index(drop=True)
d1 = d1[selected_columns].reset_index(drop=True)
d2 = d2[selected_columns].reset_index(drop=True)

df_concat_rows = pd.concat([df_single_value, d1, d2], axis=0, ignore_index=True)
df_concat_rows.to_excel(EXCEL_FOLDER + "\\raw_output.xlsx", index=False)

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
# train_df.to_excel(EXCEL_FOLDER + '\\train_data.xlsx', index=False)
# test_df.to_excel(EXCEL_FOLDER + '\\test_data.xlsx', index=False)

