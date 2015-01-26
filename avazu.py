# Kaggle competition: Avazu
# avazu.py
# Yuri M. Brovman

import csv
import pandas as pd
pd.options.display.max_rows = 200
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import operator
from sklearn import svm
from sklearn import linear_model
from sklearn import cross_validation as cv
from sklearn import metrics
from sklearn import tree
from sklearn import preprocessing
from sklearn import ensemble
from sklearn.naive_bayes import GaussianNB

def makeHist(filename, len):
    dataHist = {}
    columns = []
    with open("train.csv", 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if i == 0:
                for col in row:
                    dataHist[col] = {}
                    columns.append(col)
            elif i < len:
                if i % 1000000 == 0: print i
                for j, val in enumerate(row):
                    if j != 0 and j != 1 and j != 11 and j!= 12:
                        if j == 2: val = val[-2:]
                        if val in dataHist[columns[j]]:
                            dataHist[columns[j]][val] += 1
                        else: dataHist[columns[j]][val] = 1
            else: break

    with open(filename, 'w') as f: pickle.dump(dataHist, f)

def redef(dataHist):
    dataDict = {}
    total = sum(dataHist[dataHist.keys()[0]].values())
    for col in dataHist.keys():
        dataDict[col] = {}
        vals = sorted(dataHist[col].items(), key=operator.itemgetter(1), \
                                             reverse=True)
        per = 0.
        count = 0
        for i, val in enumerate(vals):
            if per < .95:
                dataDict[col][val[0]] = i+1
            else: break
            per += float(val[1])/total
        # print col, i+1, float(i+1)/10000, "%"
    return dataDict

def makeData(filename, len, f):
    id = []
    Y = []
    X = []
    columns = []
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if i == 0:
                for col in row: columns.append(col)
            elif i < len:
                sample = row
                for j, val in enumerate(row):
                    if j != 0 and j != 1-f:
                        # if j == 2-f: val = val[-2:]
                        if val in dataDict[columns[j]]:
                            sample[j] = dataDict[columns[j]][val]
                        else: sample[j] = 0
                if f == 1: id.append(sample[0])
                if f == 0: Y.append(int(sample[1]))
                X.append(sample[2-f:])
            else: break
    return np.array(id), np.array(Y), np.delete(np.array(X), [9-f,10-f], 1)

def getLogLoss(yCV, pred):
    # using vecorized implementation
    res = np.dot(1-yCV,np.log(pred[:,0])) + np.dot(yCV,np.log(pred[:,1]))
    return -1./len(yCV) * res

def writeTest():
    print "loading..."

    idtest, Ytest, Xtest = makeData("test.csv", 5000001, 1)
    with open('gbtALL_hist1full.pickle') as f: clf = pickle.load(f)
    Xtemp = preprocessing.scale(Xtest.astype(float))
    print "predicting..."
    pred_prob_raw = clf.predict_proba(Xtemp)
    pred_prob = []
    threshold = 1e-7
    for i, val in enumerate(pred_prob_raw):
        c0 = val[0]
        c1 = val[1]
        if c0 < threshold: c0 = threshold
        if c1 < threshold: c1 = threshold
        pred_prob.append([c0,c1])

    # SAVE THE DATA TO .csv FILE
    print "saving..."
    header = [['id', 'click']]
    predTestwrite = []
    for i, val in enumerate(pred_prob):
        predTestwrite.append([idtest[i], float(val[1])])

    with open('fifth.csv', 'wb') as fp:
        write = csv.writer(fp, delimiter=',')
        write.writerows(header)
        write.writerows(predTestwrite)

def train(dataDict, hist, len):
    start_time = time.time()

    idtrain, Ytrain, Xtrain = makeData("train.csv", len, 0)
    Xtemp = preprocessing.scale(Xtrain.astype(float))
    stop = .9
    X, XCV, y, yCV = cv.train_test_split(Xtemp, Ytrain, test_size=stop, \
                                            random_state=42)

    del idtrain, Ytrain, Xtrain, Xtemp

    # clf = GaussianNB()
    # clf = svm.SVC(kernel='rbf', probability=True)
    # clf = tree.DecisionTreeClassifier()
    # clf = linear_model.LogisticRegression()
    # clf = ensemble.RandomForestClassifier()
    # clf = ensemble.GradientBoostingClassifier(learning_rate = .25, verbose=1)
    clf = ensemble.GradientBoostingClassifier()
    print "fitting model...", time.time() - start_time
    clf.fit(X, y)

    print "predicting...   ", time.time() - start_time
    uCV = [] #clf.predict(XCV)
    pred_prob_raw = clf.predict_proba(XCV)

    pred_prob = []
    threshold = 1e-7
    for i, val in enumerate(pred_prob_raw):
        c0 = val[0]
        c1 = val[1]
        if c0 < threshold: c0 = threshold
        if c1 < threshold: c1 = threshold
        pred_prob.append([c0,c1])

    pred_prob = np.array(pred_prob)
    print type(clf).__name__, " >>>", hist, "<<<    # train =", stop*(len-1)
    print 'LogLoss: ', getLogLoss(yCV, pred_prob)
    print 'CV Set Accuracy: ', clf.score(XCV, yCV)

    del X, XCV, y, yCV, uCV, pred_prob_raw, pred_prob
    return clf

###################################################################
######################## MAIN CODE HERE ###########################
###################################################################

hist = 'dataHist1full.pickle'
# makeHist(hist, 50000001)
with open(hist) as f: dataHist = pickle.load(f)
dataDict = redef(dataHist)

clf = train(dataDict, hist, 1000001)
# with open('gbtALL_hist1full.pickle', 'w') as f: pickle.dump(clf, f)
# writeTest()



