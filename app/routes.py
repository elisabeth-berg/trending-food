from flask import Flask, render_template, request
from app import app
#from flask_wtf import FlaskForm
#from wtforms.validators import DataRequired

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        submission = request.form
        image = {'src':"https://upload.wikimedia.org/wikipedia/commons/9/9f/Chocolate%28bgFFF%29.jpg",
                 'alt':"Chocolate"}
    return render_template('submit.html', submission=submission, image=image)
