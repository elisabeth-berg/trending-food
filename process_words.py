import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import string
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


def make_top_words(df, feature, n):
    label_list = []
    years, vectors = vectorize_all(df, feature)
    for year in (2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018):
        (X, word_list) = vectors[year]
        best_k = get_best_k(X)
        cluster_df, labels = make_one_cluster_df(X, word_list, best_k, year)
        label_list.append(labels)
        if year == 2009:
            top_words = cluster_df.iloc[0:n]
        else:
            top_words = pd.concat([top_words, cluster_df.iloc[0:n]], axis=1)
    return top_words, label_list


def vectorize_all(df, feature, split_years=True):
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
    if split_years:
        for yr in years:
            yrs[yr] = df[df['post_date'].dt.year.values == yr].index
            stems = df.iloc[yrs[yr]][feature]
            X = vectorizer.fit_transform(stems)
            features = vectorizer.get_feature_names()
            vectors[yr] = (X, features)
    else:
        stems = df[feature]
        X = vectorizer.fit_transform(stems)
        features = vectorizer.get_feature_names()
        vectors = (X, features)
    return yrs, vectors


def make_one_cluster_df(X, word_list, best_k, year):
    """
    Input
    ------
    X : numpy array, feature matrix (tfidf) for one year
    word_list : list of str, feature names corresponding to the tfidf matrix X
    best_k : int, optimal number of clusters, found using get_best_k
    year : int, year corresponding to X
    """
    cluster_df = pd.DataFrame(word_list)
    centroids, labels = make_clusters(X, word_list, best_k)
    centroids_sorted = centroids.argsort()[:,-1::-1]
    for i, c in enumerate(centroids_sorted):
        cluster_df['{}_{}'.format(year, i+1)] = np.array(word_list)[centroids_sorted[i]]
        cluster_df['s{}_{}'.format(year, i+1)] = centroids[i][centroids_sorted[i]]
    if 0 in cluster_df.columns:
        cluster_df.drop(columns=[0], inplace=True)
    return cluster_df, labels



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
    return centroids, kmeans.labels_


def get_best_k(X, maxk=20):
    silhouette = np.zeros(maxk)
    for k in range(1, maxk):
        km = KMeans(k)
        y = km.fit_predict(X)
        if k > 1:
            silhouette[k] = silhouette_score(X, y)
    best_k = np.argmax(silhouette) + 2
    return best_k


def clean_one_doc(doc):
    """
    Tokenizer function for ingredient lists, can be used as input to TfidfVectorizer.

    Input
    ------
    doc : str, a document for custom processing

    Output
    ------
    doc_stems : cleaned, stemmed, & lemmatized doc
    """
    food_stops = {'pinch', 'serving', 'teaspoon', 'teaspoons', 'tablespoon',
                  'tablespoons', 'cup', 'cups', 'taste', 'oz', 'package', 'note',
                  'cut', 'inch', 'pounds', 'pound', 'ml', 'ounce', 'ounces', 'qt', 'quart'
                  'one', 'tsp', 'tbsp', 'g', 'grams','milliliters', 'milliliter', 'liter',
                  'liters', 'pint', 'pints', 'ground', 'small', 'medium', 'large', 'size'}
    food_stops2 = {'salt', 'pepper', 'buttah', 'foodcom', 'https', 'etc', 'can', 'quality',
                   'recipe', 'recipes', 'room', 'temperature', 'seeds', 'piece', 'pieces',
                   'thick', 'thin', 'thinly', 'part', 'firm', 'favorite', 'envelope', 'envelopes'}

    porter = PorterStemmer()
    keep_chars = set(string.ascii_lowercase + ' ')
    doc = doc.lower()
    doc = re.sub(r'(-|/)', ' ', doc)
    doc = ''.join(ch for ch in doc if ch in keep_chars)
    doc = word_tokenize(doc)
    doc = [word[0] for word in pos_tag(doc) if word[1] in {'NN', 'NNS'}]
    sw = set(stopwords.words('english'))
    sw.update(food_stops)
    sw.update(food_stops2)
    doc = [word for word in doc if not word in sw]
    doc_stems = [porter.stem(word) for word in doc]
    return doc_stems
