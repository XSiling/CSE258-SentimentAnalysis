from preprocess import preprocess
import pandas as pd
import codecs
from tqdm import tqdm
import numpy as np
import wordcloud
import seaborn as sns
from nltk import ngrams
from collections import defaultdict
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import pickle
import ssl
import json
import random


if __name__ == '__main__':
    print("get training and testing set")
    with codecs.open('../training.1600000.processed.noemoticon.csv','r',encoding='latin-1') as fp:
        data = pd.read_csv(fp, header=None)
    positiveTweets = []
    positiveTweetTime = []
    negativeTweets = []
    negativeTweetTime = []
    for index, row in tqdm(data.iterrows(), desc='reading information from data'):
        sentiment = row[0]
        tweet = row[5]
        time = row[2]
        if sentiment == 0:
            negativeTweets.append(tweet)
            negativeTweetTime.append(time)
        else:
            positiveTweets.append(tweet)
            positiveTweetTime.append(time)

    preprocessedNegativeTweets = preprocess(negativeTweets)
    preprocessedPositiveTweets = preprocess(positiveTweets)
    preprocessedNegativeTweets = [' '.join(w) for w in preprocessedNegativeTweets]
    preprocessedPositiveTweets = [' '.join(w) for w in preprocessedPositiveTweets]
    # get training set and testing set

    negativeCnt, positiveCnt = len(preprocessedNegativeTweets), len(preprocessedPositiveTweets)
    
    negativeSets = [[preprocessedNegativeTweets[i], negativeTweetTime[i]] for i in range(negativeCnt)]
    positiveSets = [[preprocessedPositiveTweets[i], positiveTweetTime[i]] for i in range(positiveCnt)]

    random.shuffle(negativeSets)
    random.shuffle(positiveSets)

    assert negativeCnt == positiveCnt

    trainingLength = int(0.08*negativeCnt)
    totalLength = int(0.1 * negativeCnt)
    testingLength = totalLength - trainingLength
    training = negativeSets[0:trainingLength] + positiveSets[0:trainingLength]
    testing = negativeSets[trainingLength:totalLength] + positiveSets[trainingLength:totalLength]

    trainingSet = [{"sentence":w[0],"time":w[1], "sentiment":1} for w in training]
    for i in range(len(trainingSet)//2):
        trainingSet[i]["sentiment"] = 0
    testingSet = [{"sentence":w[0], "time":w[1], "sentiment":1} for w in testing]
    for i in range(len(testingSet)//2):
        testingSet[i]["sentiment"] = 0

    with open('train.json','w') as fp:
        json.dump(trainingSet, fp)

    with open('test.json','w') as fp:
        json.dump(testingSet, fp)

    print("Save successfully for train.json and test.json")
    # print("providing for nn dataset...")

    # trainLength = int(0.75 * len(trainingSet))
    # validLength = int(0.25 * len(trainingSet))
    # testLength = len(testingSet)

    # validingSet = trainingSet[0: validLength//2] + trainingSet[len(trainingSet)//2: len(trainingSet)//2 + validLength//2]
    # trainingSet = trainingSet[validLength//2: len(trainingSet)//2] + trainingSet[len(trainingSet)//2 + validLength//2:]
    # # testingSet
    # print("train valid test split: ")
    # print("train size:", len(trainingSet), "valid size:", len(validingSet), "test size:", len(testingSet))

    # corpus = [w['sentence'] for w in trainingSet]
    # corpus_time = [w['time'] for w in trainingSet]
    # corpus_y = [w['sentiment'] for w in trainingSet]

    # corpus_valid = [w['sentence'] for w in validingSet]
    # corpus_valid_time = [w['time'] for w in validingSet]
    # corpus_valid_y = [w['sentiment'] for w in validingSet]

    # corpus_test = [w['sentence'] for w in testingSet]
    # corpus_test_time = [w['time'] for w in testingSet]
    # corpus_test_y = [w['sentiment'] for w in testingSet]

    # weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    # months = ["Apr", "May", "Jun"]

    # for i in range(len(corpus_time)):
    #     tmp = [0 for i in range(6)] + [0 for i in range(2)] + [0 for i in range(30)] + [0 for i in range(23)]
    #     tmp_time = corpus_time[i]
    #     weekday = tmp_time.split(' ')[0]
    #     weekdayIndex = weekdays.index(weekday)
    #     if weekdayIndex < 6: tmp[weekdayIndex] = 1

    #     month = tmp_time.split(' ')[1]
    #     monthIndex = months.index(month)
    #     if monthIndex < 2: tmp[6 + monthIndex] = 1

    #     day = tmp_time.split(' ')[2]
    #     dayIndex = int(day)
    #     if dayIndex < 30: tmp[8 + dayIndex] = 1

    #     hour = tmp_time.split(' ')[3].split(':')[0]
    #     hourIndex = int(hour)
    #     if hourIndex < 23: tmp[38 + hourIndex] = 1

    #     corpus_time[i] = tmp

    # for i in range(len(corpus_valid_time)):
    #     tmp = [0 for i in range(6)] + [0 for i in range(2)] + [0 for i in range(30)] + [0 for i in range(23)]
    #     tmp_time = corpus_valid_time[i]
    #     weekday = tmp_time.split(' ')[0]
    #     weekdayIndex = weekdays.index(weekday)
    #     if weekdayIndex < 6: tmp[weekdayIndex] = 1

    #     month = tmp_time.split(' ')[1]
    #     monthIndex = months.index(month)
    #     if monthIndex < 2: tmp[6 + monthIndex] = 1

    #     day = tmp_time.split(' ')[2]
    #     dayIndex = int(day)
    #     if dayIndex < 30: tmp[8 + dayIndex] = 1

    #     hour = tmp_time.split(' ')[3].split(':')[0]
    #     hourIndex = int(hour)
    #     if hourIndex < 23: tmp[38 + hourIndex] = 1

    #     corpus_valid_time[i] = tmp
    
    # for i in range(len(corpus_test_time)):
    #     tmp = [0 for i in range(6)] + [0 for i in range(2)] + [0 for i in range(30)] + [0 for i in range(23)]
    #     tmp_time = corpus_test_time[i]
    #     weekday = tmp_time.split(' ')[0]
    #     weekdayIndex = weekdays.index(weekday)
    #     if weekdayIndex < 6: tmp[weekdayIndex] = 1

    #     month = tmp_time.split(' ')[1]
    #     monthIndex = months.index(month)
    #     if monthIndex < 2: tmp[6 + monthIndex] = 1

    #     day = tmp_time.split(' ')[2]
    #     dayIndex = int(day)
    #     if dayIndex < 30: tmp[8 + dayIndex] = 1

    #     hour = tmp_time.split(' ')[3].split(':')[0]
    #     hourIndex = int(hour)
    #     if hourIndex < 23: tmp[38 + hourIndex] = 1

    #     corpus_test_time[i] = tmp

    # # import pdb;pdb.set_trace()
    # corpus_time = np.array(corpus_time)
    # corpus_valid_time = np.array(corpus_valid_time)
    # corpus_test_time = np.array(corpus_test_time)

    # vector = TfidfVectorizer(min_df=100)
    # vector.fit(corpus+corpus_valid)

    # tfidf_matrix = vector.transform(corpus)
    # tfidf_matrix = tfidf_matrix.toarray()
    # tfidf_matrix_valid = vector.transform(corpus_valid)
    # tfidf_matrix_valid = tfidf_matrix_valid.toarray()
    # tfidf_matrix_test = vector.transform(corpus_test)
    # tfidf_matrix_test = tfidf_matrix_test.toarray()

    # X = np.concatenate((tfidf_matrix, corpus_time), axis=1)
    # X_valid = np.concatenate((tfidf_matrix_valid, corpus_valid_time), axis=1)
    # X_test = np.concatenate((tfidf_matrix_test, corpus_test_time), axis=1)

    # trainPickle = {"X":X, "y":corpus_y}
    # validPickle = {"X": X_valid, "y":corpus_valid_y}
    # testPickle = {"X":X_test, "y":corpus_test_y}

    # with open("train.pkl","wb") as fp:
    #     pickle.dump(trainPickle, fp)
    
    # with open("valid.pkl", "wb") as fp:
    #     pickle.dump(validPickle, fp)

    # with open("test.pkl", "wb") as fp:
    #     pickle.dump(testPickle, fp)
