from flask import Flask, render_template, request
from app import app
from app.plot import make_plot
from src.plotters import yearly_food_plot
from io import BytesIO
import pandas as pd
#from flask_wtf import FlaskForm
#from wtforms.validators import DataRequired

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit():
    submission = request.form
    return render_template('submit.html', submission=submission)

@app.route('/fig/<ingredient>', methods=['GET', 'POST'])
def fig(ingredient):
#    fig = make_plot(ingredient)
    df = pd.read_pickle('data/featured_recipes.pkl')
    fig = yearly_food_plot(df, ingredient)
    img = BytesIO()
    fig.savefig(img)
    return img.getvalue(), 200, {'Content-Type': 'image/png'}
