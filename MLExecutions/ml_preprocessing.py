import pandas as pd
import fasttext
import ast
from main import EXCEL_FOLDER, DATA_FOLDER, FASTTEXT_PARAM
from DataFunctions.clean_transform_data import perform_tf_idf, create_fasttext_format


train_data = pd.read_excel(EXCEL_FOLDER + '\\train_data.xlsx')

train_data['aspect'] = train_data['aspect'].apply(ast.literal_eval)
train_data['sentiment'] = train_data['sentiment'].apply(ast.literal_eval)
train_data['aspect'] = train_data['aspect'].apply(lambda x: [str(item) if isinstance(item, int) else item for item in x])
train_data['sentiment'] = train_data['sentiment'].apply(lambda x: [str(item) if isinstance(item, int) else item for item in x])

train_data['Thema'] = train_data['Thema'].apply(ast.literal_eval)
train_data['Tonalität'] = train_data['Tonalität'].apply(ast.literal_eval)

tf_idf_sum = perform_tf_idf(train_data)
train_data['tf_idf_sum'] = tf_idf_sum
train_data.to_excel(EXCEL_FOLDER + '\\train_data.xlsx')

train_file_txt = DATA_FOLDER + 'fasttext_train_data.txt'
label_columns = ['Thema', 'Tonalität']
create_fasttext_format(train_data, train_file_txt, label_columns)

fasttext_model = fasttext.train_unsupervised(**FASTTEXT_PARAM)
fasttext_model.save_model(DATA_FOLDER+"fasttext_model.bin")


a = 1
