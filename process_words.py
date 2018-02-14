from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
from nltk.stem.porter import PorterStemmer

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

    porter = PorterStemmer()
    keep_chars = set(string.ascii_lowercase + ' ')
    doc = doc.lower()
    doc = re.sub(r'(-|/)', ' ', doc)
    doc = ''.join(ch for ch in doc if ch in keep_chars)
    doc = word_tokenize(doc)
    sw = set(stopwords.words('english'))
    sw.update(food_stops)
    doc = [word for word in doc if not word in sw]
    doc_stems = [porter.stem(word) for word in doc]
    return doc


def vectorize_all(df):
    vectorizer = TfidfVectorizer(tokenizer=clean_one_doc)
    years = set(date.year for date in df.post_date)
    food_yrs = {}
    vectors = {}
    for yr in years:
        food_yrs[yr] = df[df['post_date'].dt.year.values == yr].index
        food_stems = df.iloc[food_yrs[yr]]['foods']
        X = vectorizer.fit_transform(food_stems)
        features = vectorizer.get_feature_names()
        vectors[yr] = (X, features)
    return food_yrs, vectors


def make_clusters(X, features, n_features, best_k=None):
    """
    """
    maxk = len(features)//20
    silhouette = np.zeros(maxk)
    if best_k == None:
        for k in range(1, maxk):
            km = KMeans(k)
            y = km.fit_predict(X)
            if k > 1:
                silhouette[k] = silhouette_score(X, y)
        best_k = np.argmax(silhouette) + 2

    kmeans = KMeans(n_clusters=best_k).fit(X)
    centroids = kmeans.cluster_centers_

    for i, c in enumerate(centroids):
        ind = c.argsort()[::-1][:n_features]
        print('Cluster {}'.format(i))
        for i in ind:
            print('{} || {}'.format(features[i], c[i]))
        print('----------------------')
    return centroids, silhouette
