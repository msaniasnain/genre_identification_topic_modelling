import pickle
import pandas as pd
from gensim.corpora import Dictionary


def load_model_and_data():
    lda_model_f = open("lda_model.pkl", "rb")
    lda_model = pickle.load(lda_model_f)
    lda_model_f.close()

    df = pd.read_csv('lda_result_file.csv')

    return lda_model, df


def prep_data(data):
    tokenized_summaries = [summary.lower().split() for summary in data]
    dictionary = Dictionary(tokenized_summaries)
    return dictionary


def detect_genre_lda(test_summary):

    model, df = load_model_and_data()
    dictionary = prep_data(df['summary'])
    test_summary_bow = dictionary.doc2bow([each for each in test_summary.split()])
    test_topics = model.get_document_topics(test_summary_bow)
    max_topic = max(test_topics, key=lambda item: item[1])[0]

    test_df = df[df['Dominant Topic'] == max_topic]

    genre = test_df['genre'].unique()[0]
    movies = test_df.iloc[0:5]['title'].to_list()
    titles = [each.split('.')[1].strip() for each in movies]
    return genre, titles
