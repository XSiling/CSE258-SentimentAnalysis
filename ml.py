import json
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from nltk import MaxentClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

if __name__ == '__main__':
    with open('train.json', 'r')as fp:
        train = json.load(fp)

    with open('test.json', 'r') as fp:
        test = json.load(fp)

    corpus = [w['sentence'] for w in train]
    corpus_time = [w['time'] for w in train]
    corpus_y = [w['sentiment'] for w in train]
    corpus_test = [w['sentence'] for w in test]
    corpus_test_time = [w['time'] for w in test]
    corpus_test_y = [w['sentiment'] for w in test]

    # transform time
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    months = ["Apr", "May", "Jun"]

    for i in range(len(corpus_time)):
        tmp = [0 for i in range(6)] + [0 for i in range(2)] + [0 for i in range(30)] + [0 for i in range(23)]
        tmp_time = corpus_time[i]
        weekday = tmp_time.split(' ')[0]
        weekdayIndex = weekdays.index(weekday)
        if weekdayIndex < 6: tmp[weekdayIndex] = 1

        month = tmp_time.split(' ')[1]
        monthIndex = months.index(month)
        if monthIndex < 2: tmp[6 + monthIndex] = 1

        day = tmp_time.split(' ')[2]
        dayIndex = int(day)
        if dayIndex < 30: tmp[8 + dayIndex] = 1

        hour = tmp_time.split(' ')[3].split(':')[0]
        hourIndex = int(hour)
        if hourIndex < 23: tmp[38 + hourIndex] = 1

        corpus_time[i] = tmp


    for i in range(len(corpus_test_time)):
        tmp = [0 for i in range(6)] + [0 for i in range(2)] + [0 for i in range(30)] + [0 for i in range(23)]
        tmp_time = corpus_test_time[i]
        weekday = tmp_time.split(' ')[0]
        weekdayIndex = weekdays.index(weekday)
        if weekdayIndex < 6: tmp[weekdayIndex] = 1

        month = tmp_time.split(' ')[1]
        monthIndex = months.index(month)
        if monthIndex < 2: tmp[6 + monthIndex] = 1

        day = tmp_time.split(' ')[2]
        dayIndex = int(day)
        if dayIndex < 30: tmp[8 + dayIndex] = 1

        hour = tmp_time.split(' ')[3].split(':')[0]
        hourIndex = int(hour)
        if hourIndex < 23: tmp[38 + hourIndex] = 1

        corpus_test_time[i] = tmp

    corpus_time = np.array(corpus_time)
    corpus_test_time = np.array(corpus_test_time)

    # transform content
    vector = TfidfVectorizer(min_df=100)
    tfidf_matrix = vector.fit_transform(corpus)
    tfidf_matrix = tfidf_matrix.toarray()

    tfidf_matrix_test = vector.transform(corpus_test)
    tfidf_matrix_test = tfidf_matrix_test.toarray()

    X = np.concatenate((tfidf_matrix, corpus_time), axis=1)
    X_test = np.concatenate((tfidf_matrix_test, corpus_test_time), axis=1)
    
    print("dataset prepared, training dataset size:", X.shape[0], " testing set size:", X_test.shape[0])

    models = [GaussianNB(), RandomForestClassifier(), svm.SVC(), DecisionTreeClassifier()]
    modelNames = ["Naive Bayesian", "Random Forest", "SVM", "Decision Tree"]
    for i in tqdm(range(len(models))):
        model = models[i]
        model.fit(X, corpus_y)
        y_pred = model.predict(X_test)
        acc = accuracy_score(corpus_test_y, y_pred)
        precision = precision_score(corpus_test_y, y_pred)
        recall = recall_score(corpus_test_y, y_pred)

        print(modelNames[i], "accuracy:", acc, " precision:", precision, " recall:", recall)