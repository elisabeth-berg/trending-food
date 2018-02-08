import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests


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
    foods = [food.text.strip() for food in ingredients]
    return (date, title, foods)
