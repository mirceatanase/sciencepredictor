from tfidfvec import read_object
from tfidfvec import write_object
from tfidfvec import load_dataset
from tfidfvec import StemmedTfidfVectorizer
from sklearn.pipeline import Pipeline
import numpy as np
import os

def load_model():
	path = os.path.join("data", "tfidf")
	tfidf = read_object(path)

	path = os.path.join("data", "lsa")
	lsa = read_object(path)

	path = os.path.join("data", "clf")
	clf = read_object(path)


	tfidf.set_params(input="content")
	clf = Pipeline(
		steps=[
		    ('vectorize', tfidf),
		    ('reduction', lsa),
		    ('classify', clf)
		]
	)

	return clf
