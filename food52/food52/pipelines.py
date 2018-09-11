# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

from datetime import datetime
import re
import string

from sqlalchemy.orm import sessionmaker
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

from models import RecipeModel, db_connect, create_recipes_table


class CleanseRecipePipeline(object):

    def process_item(self, item, spider):
        item['title'] = item['title'].lower().replace('\xa0', ' ')
        item['author'] = item['author'].lower()
        item['date'] = datetime.strptime(item['date'], '%B %d, %Y')
        item['n_comments'] = int(item['n_comments'])
        if item['rating'] == '':
            item['rating'] = None
        else:
            item['rating'] = float(item['rating'])
        if item['n_ratings'] == '':
            item['n_ratings'] = 0
        else:
            item['n_ratings'] = int(item.get('n_ratings', 0))
        item['n_faves'] = int(item['n_faves'])
        item['ingredients'] = clean_one_doc(item['ingredients'])
        item['tags'] = [tag.lower() for tag in item['tags']]
        return item

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
    doc = [word[0] for word in pos_tag(doc) if word[1] in {'NN', 'NNS', 'JJ'}]
    sw = set(stopwords.words('english'))
    sw.update(food_stops)
#    sw.update(food_stops2)
    doc = [word for word in doc if not word in sw]
    doc_stems = [porter.stem(word) for word in doc]
    return doc_stems


class SaveRecipePipeline(object):

    def __init__(self):
        """
        Initializes database connection and sessionmaker.
        Creates recipes table.
        """
        engine = db_connect()
        create_recipes_table(engine)
        self.Session = sessionmaker(bind=engine)

    def process_item(self, item, spider):
        """
        Save recipe into the database.
        This method is called for every item pipeline component.
        """
        session = self.Session()
        recipe = RecipeModel(**item)
        try:
            session.add(recipe)
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()
        return item
