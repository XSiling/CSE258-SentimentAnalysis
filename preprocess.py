import re
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pdb

def preprocess(sentences):
    # step1 - segmentation
    print("step1: remove abnormal signs and segmentation")
    sentences0 = []
    for sentence in sentences:
        newSentence = re.sub(r'[{}]+'.format(punctuation), '', sentence)
        # pdb.set_trace()
        sentences0.append(newSentence)
    
    # cleaning
    print("step2: remove stop words...")
    # pdb.set_trace()
    stopWords = set(stopwords.words('english'))
    sentences1 = []
    for sentence in sentences0:
        try:
            newSentence = [w.lower() for w in sentence.split(' ') if w not in stopWords]
            sentences1.append(newSentence)
        except:
            print(sentence)

    # stemming & lemmatization
    print("step3: lemmatization...")
    wnl = WordNetLemmatizer()
    sentences2 = []
    for sentence in sentences1:
        newSentence = [wnl.lemmatize(w) for w in sentence]
        sentences2.append(newSentence)

    # final cleaning
    print("step4: final cleaning")
    sentences3 = []
    for sentence in sentences2:
        newSentence = [w for w in sentence if w != '']
        sentences3.append(newSentence)
    
    return sentences3

