from flask import Flask, render_template, request, jsonify
from app import app
from app.plot import make_plot
from src.plotters import yearly_food_plot, time_food_plot
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


@app.route('/fig1/<ingredient>', methods=['GET'])
def fig1(ingredient):
    fig = yearly_food_plot(df, ingredient, all_plots=True)
    img = BytesIO()
    fig.savefig(img)
    return img.getvalue(), 200, {'Content-Type': 'image/png'}


@app.route('/fig2/<ingredient>', methods=['GET'])
def fig2(ingredient):
    fig = time_food_plot(df, ingredient, n_months=3)
    img = BytesIO()
    fig.savefig(img)
    return img.getvalue(), 200, {'Content-Type': 'image/png'}


@app.route('/recommend/<ingredient>', methods=['GET'])
def recommend(ingredient):
    pairings = you_might_like(full_G, ingredient, 15)
    keys = range(len(pairings))
    d = dict(zip(keys, pairings))
    return jsonify(d)
