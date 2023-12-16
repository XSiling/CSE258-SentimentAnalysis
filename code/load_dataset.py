import json
import random
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm
import pickle

def load_dataset(train_path, test_path, model, includeTime=True, validPercent=0.2):
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    months = ["Apr", "May", "Jun"]

    with open(train_path,'r') as fp:
        trainData = json.load(fp)
    with open(test_path, 'r') as fp:
        testData = json.load(fp)
    
    random.shuffle(trainData)
    validData = trainData[0:int(len(trainData)*validPercent)]
    trainData = trainData[int(len(trainData)*validPercent):]

    train_y = [trainData[i]['sentiment'] for i in range(len(trainData))]
    valid_y = [validData[i]['sentiment'] for i in range(len(validData))]
    test_y = [testData[i]['sentiment'] for i in range(len(testData))]

    word2vec = Word2Vec.load(model)
    # train
    x_train = []
    for i in tqdm(range(len(trainData)), desc='loading train'):
        sentence = trainData[i]['sentence']
        words = sentence.split(' ')
        vectors = [word2vec.wv[word] for word in words if word in word2vec.wv.key_to_index.keys()]
        if len(vectors) != 0:
            x = sum(vectors)/len(vectors)
            x = x.reshape(1, -1)
        else:
            x = np.zeros((1, 300))
        x_train.append(x)
    x_train = np.array(x_train).squeeze()
    
    if includeTime:
        x_train_time = []
        for i in range(len(trainData)):
            tmp = [0 for i in range(6)] + [0 for i in range(2)] + [0 for i in range(30)] + [0 for i in range(23)]
            tmp_time = trainData[i]['time']
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

            x_train_time.append(tmp)
        x_train_time = np.array(x_train_time)
        x_train = np.concatenate((x_train, x_train_time), axis=1)        

    print("training data shape:", x_train.shape)


    # valid
    x_valid = []
    for i in tqdm(range(len(validData)), desc='loading valid'):
        sentence = validData[i]['sentence']
        words = sentence.split(' ')
        vectors = [word2vec.wv[word] for word in words if word in word2vec.wv.key_to_index.keys()]
        if len(vectors) != 0:
            x = sum(vectors)/len(vectors)
            x = x.reshape(1, -1)
        else:
            x = np.zeros((1, 300))
        x_valid.append(x)
    x_valid = np.array(x_valid).squeeze()
    
    if includeTime:
        x_valid_time = []
        for i in range(len(validData)):
            tmp = [0 for i in range(6)] + [0 for i in range(2)] + [0 for i in range(30)] + [0 for i in range(23)]
            tmp_time = validData[i]['time']
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

            x_valid_time.append(tmp)
        x_valid_time = np.array(x_valid_time)
        x_valid = np.concatenate((x_valid, x_valid_time), axis=1)      

    print("validating data shape: ", x_valid.shape)


    # test
    x_test = []
    for i in tqdm(range(len(testData)), desc='loading test'):
        sentence = testData[i]['sentence']
        words = sentence.split(' ')
        vectors = [word2vec.wv[word] for word in words if word in word2vec.wv.key_to_index.keys()]
        if len(vectors) != 0:
            x = sum(vectors)/len(vectors)
            x = x.reshape(1, -1)
        else:
            x = np.zeros((1, 300))
        x_test.append(x)
    x_test = np.array(x_test).squeeze()
    
    if includeTime:
        x_test_time = []
        for i in range(len(testData)):
            tmp = [0 for i in range(6)] + [0 for i in range(2)] + [0 for i in range(30)] + [0 for i in range(23)]
            tmp_time = testData[i]['time']
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

            x_test_time.append(tmp)
        x_test_time = np.array(x_test_time)
        x_test = np.concatenate((x_test, x_test_time), axis=1)

    print("testing data shape: ", x_test.shape)

    trainingSet = {"X": x_train, "y": train_y}
    validatingSet = {"X":x_valid, "y": valid_y}
    testingSet = {"X":x_test, "y": test_y}

    with open('train-ablation.pkl', 'wb') as fp:
        pickle.dump(trainingSet, fp)

    with open('valid-ablation.pkl', 'wb') as fp:
        pickle.dump(validatingSet, fp)

    with open('test-ablation.pkl', 'wb') as fp:
        pickle.dump(testingSet, fp)
    
    return trainingSet, validatingSet, testingSet

if __name__ == '__main__':
    load_dataset('train.json', 'test.json', 'word2vec.model', True, 0.2)
