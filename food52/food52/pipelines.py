# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

from datetime import datetime

from sqlalchemy.orm import sessionmaker
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
        return item

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
