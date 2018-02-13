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

def clean_all(docs):
    """
    """
    pass
