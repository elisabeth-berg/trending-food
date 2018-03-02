from flask import Flask, render_template, request
from app import app
from app.plot import make_plot
from src.plotters import yearly_food_plot
from io import BytesIO
import pandas as pd
df = pd.read_pickle('data/featured_recipes.pkl')

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
