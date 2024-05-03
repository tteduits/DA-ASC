import spacy
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from statistics import mean
from tqdm import tqdm
import pickle
from main import DATA_FOLDER
nlp = spacy.load(r'C:\Users\tijst\AppData\Roaming\Python\Python38\site-packages\de_core_news_sm\de_core_news_sm-3.7.0')


def remove_duplicate_themes(df):

    for idx, row in df.iterrows():
        theme_count = {}
        for i, theme in enumerate(row['Thema']):
            if theme in theme_count:
                theme_count[theme].append(i)
            else:
                theme_count[theme] = [i]

        for theme, indices in theme_count.items():
            if len(indices) > 1:

                remove_idx = indices[1]
                del row['Thema'][remove_idx]
                del row['sentiment'][remove_idx]
                del row['aspect'][remove_idx]
                del row['Tonalität'][remove_idx]

    df.reset_index()
    return df


def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_tokens = []

    for token in doc:
        if token.is_punct:
            lemmatized_tokens.append(token.text)
        else:
            lemmatized_tokens.append(token.lemma_)

    lemma_text = ' '.join(lemmatized_tokens)
    lemma_text = lemma_text.lower()

    return lemma_text


def tokenize_sentences(text):
    return sent_tokenize(text)


def calculate_tf_idf_scores(corpus):
    nltk.download('stopwords')
    german_stop_words = stopwords.words('german')
    vectorizer = TfidfVectorizer(stop_words=german_stop_words)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    with open(DATA_FOLDER+"\\tfidf_summarizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    return tfidf_matrix, vectorizer.get_feature_names_out()


def calculate_sentence_scores(tf_idf_matrix):
    sentence_scores = {}
    num_sentences = tf_idf_matrix.shape[0]  # Number of sentences in the document
    for i in range(num_sentences):
        sentence_scores[i] = np.mean(tf_idf_matrix[i].toarray())
    return sentence_scores


def find_average_score(scores):
    return mean(scores.values())


def generate_summary(sentences, scores, threshold):
    summary = ""
    for i, sentence in enumerate(sentences):
        if i in scores and scores[i] >= threshold:
            summary += sentence + " "
    return summary


def perform_tf_idf(data):
    corpus = data['lemma_text']
    tfidf_matrix, feature_names = calculate_tf_idf_scores(corpus)
    avg_score = find_average_score(calculate_sentence_scores(tfidf_matrix))
    threshold = 0.9 * avg_score

    summaries = []
    for doc_index, doc_text in tqdm(enumerate(corpus), desc="Making summaries", total=len(corpus)):
        sentences = tokenize_sentences(doc_text)
        sentence_scores = calculate_sentence_scores(tfidf_matrix[:len(sentences)])  # Update to handle variable-length documents
        summary = generate_summary(sentences, sentence_scores, threshold)
        summaries.append(summary)

    return summaries


def create_fasttext_format(df, output_file, label_columns):
    with open(output_file, 'w', encoding='utf-8') as f:
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Creating word embeddings"):
            # Labels are saved as '__label__ThemeColValue __label__SentimentColValue'
            labels_str = '__label__' + 'Aspect_' + '_'.join(row['aspect']) + '_'
            labels_str = labels_str + 'Sentiment_' + '_'.join(row['sentiment'])

            line = labels_str + ' ' + row['tf_idf_sum'] + '\n'
            f.write(line)
