import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

def clean_one_doc(doc):
    """
    Input
    ------
    doc : str, a document for custom processing

    Output
    ------
    doc_stems : cleaned, stemmed, & lemmatized doc
    """
    food_stops = {'pinch', 'serving', 'teaspoon', 'teaspoons', 'tablespoon',
                  'tablespoons', 'cup', 'cups', 'optional', 'taste', 'oz',
                  'package', 'see', 'note', 'small', 'medium', 'large', 'cut',
                  'inch', 'divided', 'pounds', 'pound', 'plus', 'ml', 'thinly',
                  'sliced', 'finely', 'chopped', 'dried', 'ounce', 'ounces',
                  'grated', 'one', 'minced', 'tsp', 'tbsp', 'rinsed', 'like',
                  'g', 'grams', 'washed', 'milliliters', 'milliliter', 'liter',
                  'liters', 'pint', 'pints'}
    food_stops2 = {'salt', 'pepper', 'fresh', 'freshly',
                   'ground', 'extra', 'needed', 'buttah', 'foodcom', 'https',
                   'recipe', 'recipes', 'room', 'temperature', 'peeled', 'seeds'}

    porter = PorterStemmer()
    keep_chars = set(string.ascii_lowercase + ' ')
    doc = doc.lower()
    doc = re.sub(r'(-|/)', ' ', doc)
    doc = ''.join(ch for ch in doc if ch in keep_chars)
    doc = word_tokenize(doc)
    sw = set(stopwords.words('english'))
    sw.update(food_stops)
    sw.update(food_stops2)
    doc = [word for word in doc if not word in sw]
    doc_stems = [porter.stem(word) for word in doc]
    return doc


def vectorize_all(df, feature):
    """
    Input
    ------
    df : Pandas dataframe with 'post_date' column
    feature : name of column on which to perform tfidf

    Output
    ------
    yrs : dict of form {year : index_list}, containing the indices of all
          rows corresponding to that year
    vectors : dict of form {year : (tfidf_matrix, word_list)},
          where tfidf_matrix is a sparse matrix for the features in this
          year, and word_list is the corpus of words.
    """
    vectorizer = TfidfVectorizer(tokenizer=clean_one_doc)
    years = set(date.year for date in df.post_date)
    yrs = {}
    vectors = {}
    for yr in years:
        yrs[yr] = df[df['post_date'].dt.year.values == yr].index
        stems = df.iloc[yrs[yr]][feature]
        X = vectorizer.fit_transform(stems)
        features = vectorizer.get_feature_names()
        vectors[yr] = (X, features)
    return yrs, vectors


def get_best_k(X, maxk=20):
    silhouette = np.zeros(maxk)
    for k in range(1, maxk):
        km = KMeans(k)
        y = km.fit_predict(X)
        if k > 1:
            silhouette[k] = silhouette_score(X, y)
    best_k = np.argmax(silhouette) + 2
    return best_k, silhouette


def make_clusters(X, features, best_k, n_words=10, print=False):
    """
    Cluster the documents in X using KMeans clustering
    Print the top words for each cluster

    Input
    ------
    X : matrix of tfidf vectors (output of vectorize_all)
    features : list of words corresponding to the columns of X
               (output of vectorize_all)
    n_words :  number of words to display for each cluster when printing
    best_k :   int, a previously determined number of clusters

    Output
    ------
    centroids : cluster centers from KMeans
    """
    kmeans = KMeans(n_clusters=best_k).fit(X)
    centroids = kmeans.cluster_centers_
    if print:
        for i, c in enumerate(centroids):
            ind = c.argsort()[::-1][:n_words]
            print('Cluster {}'.format(i))
            for i in ind:
                print('{} || {}'.format(features[i], c[i]))
            print('----------------------')
    return centroids
