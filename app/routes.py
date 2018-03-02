from flask import Flask, render_template, request, jsonify
from app import app
from app.plot import make_plot
from src.plotters import yearly_food_plot
from src.process_words import you_might_like
from io import BytesIO
import pandas as pd
import networkx as nx
df = pd.read_pickle('data/featured_recipes.pkl')
full_G = nx.read_gpickle('data/food_graph.gpickle')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/fig/<ingredient>', methods=['GET'])
def fig(ingredient):
    fig = yearly_food_plot(df, ingredient, all_plots=False)
    img = BytesIO()
    fig.savefig(img)
    return img.getvalue(), 200, {'Content-Type': 'image/png'}


@app.route('/recommend/<ingredient>', methods=['GET','POST'])
def recommend(ingredient):
    pairings = you_might_like(full_G, ingredient, 10)
    print(pairings)
    return pairings[0], 200
