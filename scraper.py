import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
import psycopg2
import time


def store_all_types(clear=False):
    conn = psycopg2.connect("dbname=food_db")
    cur = conn.cursor()
    conn.autocommit = True
    if clear:
        cur.execute("""DELETE FROM recipes_all;""")
    for r_type in ('appetizers', 'booze-free-drinks', 'bread-rolls-muffins',
                    'breakfast-brunch', 'cocktails', 'condiments', 'desserts',
                    'entrees', 'salads', 'side-dishes', 'sandwiches', 'soups'):
        print("Storing {}, thank you for waiting.".format(r_type))
        store_recipes_type(r_type)



def store_recipes_type(r_type):
    """
    Query pages of recipes of given type, store the results in recipes_all.
    """
    response = requests.get("https://food52.com/recipes/{}?".format(r_type))
    soup = BeautifulSoup(response.text, "html.parser")
    pages = soup.find("div", {"class" : "pagination"})
    pg = int(pages.find_all("a")[-2].text)

    conn = psycopg2.connect("dbname=food_db")
    cur = conn.cursor()
    conn.autocommit = True
    for page in range(1, pg+1):
        store_page_data(page)
        time.sleep(5)       #wait 5 secs after every page


def store_all_data(clear=True):
    """
    Query pages from main recipe page, store the results in food_db.
    Note that the database is cleared first as the default!
    """
    conn = psycopg2.connect("dbname=food_db")
    cur = conn.cursor()
    conn.autocommit = True
    if clear:
        cur.execute("""DELETE FROM recipes;""")
    for page in range(1, 204):
        store_page_data(page)


def store_page_data(page):
    """
    Store the recipe results from a single page into food_db.
    """
    conn = psycopg2.connect("dbname=food_db")
    cur = conn.cursor()
    conn.autocommit = True
    urls = get_urls(page)
    for url in urls:
        recipe_data = get_ingredients(url)
        query = """
            INSERT INTO
            recipes (post_date, title, foods)
            VALUES (%s, %s, %s);
            """
        cur.execute(query, recipe_data)
    conn.close()


def get_ingredients(href):
    '''
    INPUT
    ------
    href: href for a Food52 recipe's permalink

    OUTPUT
    ------
    A tuple containing
    date: pd Timestamp, the date of the posting
    title: str, the title of the recipe
    foods: list of strings, the recipe ingredients
    '''
    response = requests.get("https://food52.com/" + href)
    soup = BeautifulSoup(response.text, "html.parser")
    ingredients = soup.find_all("span", {"class" : "recipe-list-item-name"})
    title = soup.find("h1", {"class" : "article-header-title"})
    title = title.text.strip()
    title = title.replace('\xa0', ' ')
    date = soup.find("p", {'data-bind' :"with: comments"}).text.split("\n")[1].strip()[2:]
    date = pd.to_datetime(date)
    foods = {food.text.strip().lower() for food in ingredients}
    foods = ' '.join(foods)
    return (date, title, foods)



def get_urls(page):
    '''
    INPUT: page number (integer between 1 and 203)
    OUTPUT: list of urls of recipes on this page

    Find the permalink for each recipe on a given page of https://food52.com
    '''
    response = requests.get("https://food52.com/recipes?page={}".format(page))
    soup = BeautifulSoup(response.text, "html.parser")
    posts = soup.find_all("div", {"class" : "photo-block"})
    urls = [post.find('a')['href'] for post in posts]
    return urls


def get_urls_type(r_type, page):
    '''
    INPUT: page number
    OUTPUT: list of urls of recipes on this page

    Find the permalink for each recipe on a given page of https://food52.com
    '''
    response = requests.get("https://food52.com/{}?page={}".format(r_type, page))
    soup = BeautifulSoup(response.text, "html.parser")
    posts = soup.find_all("div", {"class" : "photo-block"})
    urls = [post.find('a')['href'] for post in posts]
    return urls
