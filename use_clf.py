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

clf = load_model()

content = [
    "singular value decomposition is a method for dimensionality reduction",
    "Intel processors have at least 8 cores",
    "Napoleon was a French military leader",
    "Volcano is still active in Hawaii",
    "The brain is the most important human organ",
    "mean squared error is a good approach to fit a model"
]

y = [0, 4, 3, 2, 1, 0]

y_pred = clf.predict(content)
acc = np.mean(y_pred == y)
#print(acc)


path = os.path.join("data", "clf_pipeline")
write_object(clf, path)

######################################
#################### USAGE ###########
#####################################
path = os.path.join("data", "clf_pipeline")
clf = read_object(path)

y_pred = clf.predict(content)
acc = np.mean(y_pred == y)
#print(acc)


#####################################
####################################

path = os.path.join("data", "test_lsa")
X_test, y_test, f_test = load_dataset(path)
