from flask import Flask, render_template, request
from app import app
from app.plot import make_plot
from io import BytesIO
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
    fig = make_plot(ingredient)
    img = BytesIO()
    fig.savefig(img)
    return img.getvalue(), 200, {'Content-Type': 'image/png'}
#    img.seek(0)
#    return send_file(img, mimetype='image/png')
