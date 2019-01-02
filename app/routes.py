from flask import render_template, flash, redirect, request
from app import app
from app.forms import PredictForm
import os

from use_clf import *
clf = load_model()
classes = ("mathematics", "biology", "geography", "history", "computer science")

from datetime import datetime
from elasticsearch import Elasticsearch
es = Elasticsearch()


import uuid

ALLOWED_EXTENSIONS = set(['txt'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    pred_form = PredictForm()

    if pred_form.validate_on_submit():
        flash('Detection requested')
        with open('tmp.txt', 'w') as f:
            f.write(pred_form.science_text.data)

        return redirect('/detect')

    
    return render_template('index.html', form=pred_form)

@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            flash('No file part')
            return redirect('/index')
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            print('No selected file')
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            file.save('tmp.txt')
            return redirect('/detect')


@app.route('/detect', methods=['GET', 'POST'])
def detect():
    # Predict on the text 
    with open('tmp.txt', 'r') as f:
        text = ''
        for line in f.readlines():
            text = text + line

        content = [text]
        result = clf.predict(content)
        print('Result: ', result)    
    with open('res.txt', 'w') as f:
        f.write(classes[result[0]])

    results = {'area': classes[result[0]].upper()}
    return render_template('detect.html', results=results, data=text)

@app.route('/mispredict', methods=['GET', 'POST'])
def mispredict():
    # Predict on the text 
    with open('tmp.txt', 'r') as f:
        text = ''
        for line in f.readlines():
            text = text + line

    with open('res.txt', 'r') as f:
        prediction = f.readlines()[0]

    disabled_dict = {
        'geography' : 'blue-2',
        'computer science': 'blue-2',
        'mathematics': 'blue-2',
        'biology': 'blue-2',
        'history': 'blue-2'
    }

    disabled_dict[prediction] = 'red'

    print(disabled_dict)

    return render_template('mispredict.html', data=text, disabled_dict=disabled_dict)

@app.route('/mispredict0', methods=['GET', 'POST'])
def mispredict1():
    # The text is in 'tmp.txt', the prediction is in 'res.txt'
    # And the class selected by the user is 0
    with open('tmp.txt', 'r') as f:
        text = ''
        for line in f.readlines():
            text = text + line
   
    with open('res.txt', 'r') as f:
        prediction = f.readlines()[0]    

    doc = {
        'text': text,
        'label': prediction,
        'real_label': 'mathematics',
        'timestamp': datetime.now(),
    }

    strid = str(uuid.uuid1())
    res = es.index(index="misclassified", doc_type='tweet', id=strid, body=doc)
    #print(res['result'])

    res = es.get(index="misclassified", doc_type='tweet', id=1)
    print(res['_source'])

    return redirect('/index')

@app.route('/mispredict1', methods=['GET', 'POST'])
def mispredict2():
    # The text is in 'tmp.txt', the prediction is in 'res.txt'
    # And the class selected by the user is 1
    with open('tmp.txt', 'r') as f:
        text = ''
        for line in f.readlines():
            text = text + line
   
    with open('res.txt', 'r') as f:
        prediction = f.readlines()[0]    

    doc = {
        'text': text,
        'label': prediction,
        'real_label': 'biology',
        'timestamp': datetime.now(),
    }

    res = es.index(index="misclassified", doc_type='tweet', id=1, body=doc)

    res = es.get(index="misclassified", doc_type='tweet', id=1)
    print(res['_source'])

    return redirect('/index')

@app.route('/mispredict2', methods=['GET', 'POST'])
def mispredict3():
    # The text is in 'tmp.txt', the prediction is in 'res.txt'
    # And the class selected by the user is 2
    with open('tmp.txt', 'r') as f:
        text = ''
        for line in f.readlines():
            text = text + line
   
    with open('res.txt', 'r') as f:
        prediction = f.readlines()[0]    

    doc = {
        'text': text,
        'label': prediction,
        'real_label': 'geography',
        'timestamp': datetime.now(),
    }

    res = es.index(index="misclassified", doc_type='tweet', id=1, body=doc)

    res = es.get(index="misclassified", doc_type='tweet', id=1)
    print(res['_source'])

    return redirect('/index')

@app.route('/mispredict3', methods=['GET', 'POST'])
def mispredict4():
    # The text is in 'tmp.txt', the prediction is in 'res.txt'
    # And the class selected by the user is 3
    with open('tmp.txt', 'r') as f:
        text = ''
        for line in f.readlines():
            text = text + line
   
    with open('res.txt', 'r') as f:
        prediction = f.readlines()[0]    

    doc = {
        'text': text,
        'label': prediction,
        'real_label': 'history',
        'timestamp': datetime.now(),
    }

    res = es.index(index="misclassified", doc_type='tweet', id=1, body=doc)
    #print(res['result'])

    res = es.get(index="misclassified", doc_type='tweet', id=1)
    print(res['_source'])

    return redirect('/index')

@app.route('/mispredict4', methods=['GET', 'POST'])
def mispredict5():
    # The text is in 'tmp.txt', the prediction is in 'res.txt'
    # And the class selected by the user is 4
    with open('tmp.txt', 'r') as f:
        text = ''
        for line in f.readlines():
            text = text + line
   
    with open('res.txt', 'r') as f:
        prediction = f.readlines()[0]    

    doc = {
        'text': text,
        'label': prediction,
        'real_label': 'computer science',
        'timestamp': datetime.now(),
    }

    res = es.index(index="misclassified", doc_type='tweet', id=1, body=doc)
    print(res)

    res = es.get(index="misclassified", doc_type='tweet', id=1)
    print(res['_source'])

    return redirect('/index')

