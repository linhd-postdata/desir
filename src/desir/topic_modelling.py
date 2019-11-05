# -*- coding: utf-8 -*-

import gensim
import numpy as np
import pandas as pd
import pickle
import pyLDAvis.gensim

from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from spacy.lang.es import Spanish

parser = Spanish()
stemmer = SnowballStemmer('spanish')
stop_words_es = set(stopwords.words("spanish"))
stop_words_es.update(['proyecto', 'investigacion', 'nuevo'])
tokenizer = RegexpTokenizer(r'\w+')


def tokenize(text):
    """

    :param text: to tokenize
    :return: array of tokens
    """
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


def get_lemma(word):
    """
    process of reducin a word to its word root form
    :param word:
    :return: root form of the word
    """
    return stemmer.stem(word)


def prepare_text_for_lda(text):
    """

    :param text:
    :return: list of filetered tokens
    """
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in stop_words_es]
    tokens = [get_lemma(token) for token in tokens]
    return tokens


def read_data(path):
    """
    Read xlsx file data
    :param path:
    :return: list of cells
    """
    df = pd.read_excel(path, 0)
    filtered_df = df.replace(np.nan, '-', regex=True)
    data = (filtered_df['Resumen'].tolist())
    return data


def prepare_text_data(descriptions):
    """
    Open up our data, read line by line, for each line, prepare text for LDA and
    add to a list.
    :param descriptions:
    :return: list of filetered tokens
    """
    text_data = []
    for line in descriptions:
        tokens = prepare_text_for_lda(line)
        text_data.append(tokens)
    return text_data


# LDA with Gensim
def save_lda_elements(text_data):
    """
    Create a dictionary from the data, convert to bag-of-words corpus and save
    the dictionary and corpus for future use
    :param text_data:
    :return:
    """
    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]
    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')


# Model with 5 topics
def generate_model(corpus, dictionary):
    """
    Find topics in the data
    :param corpus:
    :param dictionary:
    :return:
    """
    NUM_TOPICS = 5
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS,
                                               id2word=dictionary, passes=15)
    ldamodel.save('model5.gensim')


# Visualizing
def visualize_model():
    """
    Interactive web-based visualization of a LDA topic model
    :return:
    """
    dictionary_vis = gensim.corpora.Dictionary.load('dictionary.gensim')
    corpus_vis = pickle.load(open('corpus.pkl', 'rb'))
    lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')
    lda_display = pyLDAvis.gensim.prepare(lda, corpus_vis, dictionary_vis,
                                          sort_topics=False)
    pyLDAvis.display(lda_display)
