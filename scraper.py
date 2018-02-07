import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests


def get_urls(page):
    '''
    INPUT: page number
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
    INPUT: href for a Food52 recipe's permalink
    OUTPUT:
    '''
    response = requests.get("https://food52.com/" + href)
    soup = BeautifulSoup(response.text, "html.parser")
    ingredients = soup.find_all("span", {"class" : "recipe-list-item-name"})
    title = soup.find("h1", {"class" : "article-header-title"})
    title = title.text.strip()
    title = title.replace('\xa0', ' ')
    foods = [food.text.strip() for food in ingredients]
    return (title, foods)
