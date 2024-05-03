import numpy as np
import pandas as pd
import ast
import pickle
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from main import DATA_FOLDER


def embed_text_with_aspects(data, embedding_model, desired_embedding_dim):
    embedded_frames = []
    for index, row in data.iterrows():
        tf_idf_sum = str(row['tf_idf_sum'])  # Convert to string to handle floats
        embedding = embedding_model.get_sentence_vector(tf_idf_sum)  # Use FastText method to get embeddings
        embedding_dim = len(embedding)
        if embedding_dim < desired_embedding_dim:
            padding = np.zeros(desired_embedding_dim - embedding_dim)
            embedding = np.concatenate((embedding, padding))
        elif embedding_dim > desired_embedding_dim:
            embedding = embedding[:desired_embedding_dim]  # Truncate the embedding if it's longer

        df_row = pd.DataFrame({'embedding': [embedding], 'aspect': row['aspect'], 'sentiment': row['sentiment']})
        embedded_frames.append(df_row)

    df_embedded = pd.concat(embedded_frames, ignore_index=True)

    return df_embedded


def create_x_y_matrix_training(embedded_train_data):

    mlb = MultiLabelBinarizer()
    y_binary = pd.DataFrame(mlb.fit_transform(embedded_train_data['sentiment']), columns=mlb.classes_, index=embedded_train_data.index)

    with open(DATA_FOLDER + 'y_encoder.pkl', 'wb') as f:
        pickle.dump(mlb, f)

    mlb = MultiLabelBinarizer()
    encoded_topics = pd.DataFrame(mlb.fit_transform(embedded_train_data['aspect']), columns=mlb.classes_, index=embedded_train_data.index)
    embedding_array = np.array(embedded_train_data['embedding'].tolist())
    X_train = np.hstack((embedding_array, encoded_topics))

    with open(DATA_FOLDER + 'x_encoder.pkl', 'wb') as f:
        pickle.dump(mlb, f)

    return X_train, y_binary
