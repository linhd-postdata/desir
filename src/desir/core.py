# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from wordcloud import WordCloud

"""
nltk spanish
"""
stemmer = SnowballStemmer('spanish')
stop_words_es = set(stopwords.words("spanish"))
tokenizer = RegexpTokenizer(r'\w+')


def read_data(path):
    """
    :param path: of desir the xlsx file
    :return: String that contains all project descriptions
    """
    df = pd.read_excel(path, 0)
    filtered_df = df.replace(np.nan, '-', regex=True)
    data = (filtered_df['Resumen'].tolist())
    text = ''.join(data)
    return text


def filter_text(text):
    """
    Tokenize text and remove stop words
    :param text:
    :return: list of filtered words
    """
    tokenized_word = tokenizer.tokenize(text)
    filtered_words = [w for w in tokenized_word if
                      w.lower() not in stop_words_es]
    return filtered_words


def dist(filtered_words, num_words):
    """
    Number of the most common words to show
    :param filtered_words:
    :param num_words:
    :return: List[Tuple[_T, int]]
    """
    fdist = FreqDist(filtered_words)
    return fdist.most_common(num_words)


def plot_wordcloud(text):
    """
    Draw wordcloud with matplotlib
    :param text:
    :return:
    """
    wordcloud = WordCloud(max_font_size=50, max_words=100,
                          background_color="white",
                          relative_scaling=1.0,
                          stopwords=stop_words_es).generate(
        text)
    plt.imshow(wordcloud)
    plt.imshow(wordcloud, interpolation="bilinear")
    # bilinear is to make the displayed image appear more smoothly
    plt.axis("off")
    plt.show()
