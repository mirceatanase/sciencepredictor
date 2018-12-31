import os
import pickle
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


stemmer = nltk.stem.PorterStemmer()


class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (stemmer.stem(w) for w in analyzer(doc) if w.isalpha())


classes = ("mathematics", "biology", "geography", "history", "computer science")


def read_object(path):
    with open(path, 'rb') as fin:
        obj = pickle.load(fin)
    return obj


def write_object(obj, path):
    with open(path, 'wb') as fout:
        pickle.dump(obj, fout, pickle.HIGHEST_PROTOCOL)


def load_dataset(path):
    with open(path, 'rb') as fin:
        X, y, files = pickle.load(fin)
    return X, y, files


def store_dataset(X, y, files, path):
    with open(path, 'wb') as fout:
        pickle.dump((X, y, files), fout, pickle.HIGHEST_PROTOCOL)




if __name__ == "__main__":
    #################################################
    ############### TF IDF ##########################
    #################################################
    files = []
    y = []

    print("Loading files ...")
    for i in range(len(classes)):
        fls = os.listdir(classes[i])
        fls = [os.path.join(classes[i], file) for file in fls]
        lbs = [i] * len(fls)

        files += fls
        y += lbs

    tfidf = StemmedTfidfVectorizer(
        input="filename",
        decode_error='ignore',
        lowercase=True,
        analyzer='word',
        stop_words='english',
        token_pattern='[a-zA-Z]+',
        max_df=0.2,
        max_features=10000
    )

    print("Computing tfidf matrix ...")
    X = tfidf.fit_transform(files)#.toarray()
    y = np.array(y)

    print("Store results ...")
    path = os.path.join("data", "tfidf")
    write_object(tfidf, path)

    path = os.path.join("data", "dataset")
    store_dataset(X, y, files, path)



    #######################################################
    ############### Split Data ############################
    #######################################################
    from sklearn.model_selection import train_test_split

    path = os.path.join("data", "dataset")
    X, y, f = load_dataset(path)

    X_train, X_test, y_train, y_test, f_train, f_test = train_test_split(X, y, f, test_size=0.02)
    X_train, X_val, y_train, y_val, f_train, f_val = train_test_split(X_train, y_train, f_train, test_size=0.02)


    path = os.path.join("data", "train")
    store_dataset(X_train, y_train, f_train, path)

    path = os.path.join("data", "val")
    store_dataset(X_val, y_val, f_val, path)

    path = os.path.join("data", "test")
    store_dataset(X_test, y_test, f_test, path)



    ########################################################
    ################### DIMENSIONALITY REDUCTION ###########
    ########################################################
    from sklearn.decomposition import TruncatedSVD

    # read train data
    path = os.path.join("data", "train")
    X_train, y_train, f_train = load_dataset(path)

    # perform lsa on train data (keep 90%)
    lsa = TruncatedSVD(n_components=5000)
    X_train_lsa = lsa.fit_transform(X_train)

    print("Explained variance", lsa.explained_variance_ratio_.sum())

    # store lsa dimensionality reduction for train
    path = os.path.join("data", "train_lsa")
    store_dataset(X_train_lsa, y_train, f_train, path)

    # read val data
    path = os.path.join("data", "val")
    X_val, y_val, f_val = load_dataset(path)

    # dimensionality reduction for validation
    X_val_lsa = lsa.transform(X_val)

    # store lsa dimensionality reduction for val
    path = os.path.join("data", "val_lsa")
    store_dataset(X_val_lsa, y_val, f_val, path)


    # read test data
    path = os.path.join("data", "test")
    X_test, y_test, f_test = load_dataset(path)

    # dimnsionality reduction for test
    X_test_lsa = lsa.transform(X_test)

    # store lsa dimensionality reduction for test
    path = os.path.join("data", "test_lsa")
    store_dataset(X_test_lsa, y_test, f_test, path)

    # store lsa
    path = os.path.join("data", "lsa")
    write_object(lsa, path)

    ######################################################
    ############## SVM ###################################
    #####################################################

    path = os.path.join("data", "train_lsa")
    X_train, y_train, f_train = load_dataset(path)

    path = os.path.join("data", "val_lsa")
    X_val, y_val, f_val = load_dataset(path)

    path = os.path.join("data", "test_lsa")
    X_test, y_test, f_test = load_dataset(path)

    from sklearn import svm

    # train svm
    clf = svm.LinearSVC(C=0.05)
    clf.fit(X_train, y_train)

    # check accuracy on train
    y_pred = clf.predict(X_train)
    acc = np.mean(y_pred == y_train)
    print("Accuracy on training set", acc)


    # check accuracy on validation
    y_pred = clf.predict(X_val)
    acc = np.mean(y_pred == y_val)
    print("Accuracy on validation set", acc)


    # check accuracy on test
    y_pred = clf.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print("Accuracy on test set", acc)

    path = os.path.join("data", "clf")
    write_object(clf, path)

