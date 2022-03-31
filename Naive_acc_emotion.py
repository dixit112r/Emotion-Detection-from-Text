# -*- coding: utf-8
"""
@author: dixit
"""


# coding: utf-8

# In[1]:

from __future__ import division
import nltk 
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import *
from ourclassifier import NaiveBayes
import pickle



# In[2]:

'''
Reading the Dataset (ISEAR Dataset)
'''
Data = pd.read_csv('ISEAR.csv',header=None)
'''
0 - Class Label
1 - Sentence
'''


# In[3]:

'''
Emotion Labels
'''
emotion_labels = ['joy', 'fear', 'anger', 'sadness', 'disgust', 'shame', 'guilt']


# In[5]:

'''
Returns a list of all corresponding class labels
'''
def class_labels(emotions):
    labels = []
    labelset = []
    for e in emotions:
        labels.append(e)
        labelset.append([e])
    return labels, labelset


# In[6]:

'''
Removes unnecessary characters from sentences
'''
def removal(sentences):
    sentence_list = []
    for sen in sentences:
#         print count
#         print sen
#         print type(sen)
        s = nltk.word_tokenize(sen)
        characters = ["รก", "\xc3", "\xa1", "\n", ",", "."]
        new = ' '.join([i for i in s if not [e for e in characters if e in i]])
        sentence_list.append(new)
    return sentence_list


# In[7]:

'''
POS-TAGGER, returns NAVA words
'''
def pos_tag(sentences):
    tags = [] #have the pos tag included
    nava_sen = []
    for s in sentences:
        s_token = nltk.word_tokenize(s)
        pt = nltk.pos_tag(s_token)
        nava = []
        nava_words = []
        for t in pt:
            if t[1].startswith('NN') or t[1].startswith('JJ') or t[1].startswith('VB') or t[1].startswith('RB'):
                nava.append(t)
                nava_words.append(t[0])
        tags.append(nava)
        nava_sen.append(nava_words)
    return tags, nava_sen


# In[8]:

'''
Performs stemming
'''
def stemming(sentences):
    sentence_list = []
    sen_string = []
    sen_token = []
    stemmer = PorterStemmer()
#     i = 0
    for sen in sentences:
#         print i,
#         i += 1
        st = ""
        for word in sen:
            word_l = word.lower()
            if len(word_l) >= 3:
                st += stemmer.stem(word_l) + " "
        sen_string.append(st)
        w_set = nltk.word_tokenize(st)
        sen_token.append(w_set)
        w_text = nltk.Text(w_set)
        sentence_list.append(w_text)
    return sentence_list, sen_string, sen_token


# In[9]:

'''
Write to file
'''
def write_to_file(filename, text):
    o = open(filename,'w')
    o.write(str(text))
    o.close()


# In[10]:

'''
Creating the dataframe
'''
def create_frame(Data):
    emotions = Data[0]
    sit = Data[1]
    labels, labelset = class_labels(emotions[1:])
    sent = removal(sit[1:])
    nava, sent_pt = pos_tag(sent)
    sentences, sen_string, sen_token = stemming(sent_pt)
    frame = pd.DataFrame({0 : labels,
                          1 : sentences,
                          2 : sen_string,
                          3 : sen_token,
                          4 : labelset})
    return frame


# In[11]:

c = create_frame(Data)


# In[12]:

'''
Reads the emotion representative words file
'''
def readfile(filename):
    f = open(filename,'r')
    representative_words = []
    for line in f.readlines():
        characters = ["\n", " ", "\r", "\t"]
        new = ''.join([i for i in line if not [e for e in characters if e in i]])
        representative_words.append(new)
    return representative_words


# In[16]:

'''
Create dataset for nltk Naive Bayes
'''

def create_data(sentence, emotion):
    data = []
    for i in range(len(sentence)):
        sen = []
        for s in sentence[i]:
            sen.append(str(s))
        emo = emotion[i]
        data.append((sen, emo))
    return data

# In[17]:

'''
Get all words in dataset
'''
def get_words_in_dataset(dataset):
    all_words = []
    for (words, sentiment) in dataset:
        all_words.extend(words)
    return all_words

# In[18]
'''
Getting frequency dist of words
'''
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features



# In[19]:

'''
Extacting features
'''
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


# In[20]:

'''
Create test data
'''

def create_test(sentence, emotion):
    data = []
    sen = []
    emo = []
    for s in sentence:
        sen.append(str(s))
    for e in emotion:
        emo.append(e)
    for i in range(len(sen)):
        temp = []
        temp.append(sen[i])
        temp.append(emo[i])
        data.append(temp)
    return data




# In[21]:

'''
Classifier
'''
def classify_dataset(data):
    return         classifier.classify(extract_features(nltk.word_tokenize(data)))


# In[22]:

'''
Get accuracy
'''
def get_accuracy(test_data, classifier,emotion_labels):
    total = accuracy = float(len(test_data))
    count_emotion_result=np.zeros(len(emotion_labels))
    count_emotion_label=np.zeros(len(emotion_labels))
    labelmap = dict()
    
    for i in range(len(emotion_labels)):
        labelmap[emotion_labels[i]]=i;
    for data in test_data:
        emotion_detected = classify_dataset(data[0])
        if emotion_detected != data[1]:
            accuracy -= 1
            count_emotion_label[labelmap[data[1]]]+=1
        else:
            count_emotion_result[labelmap[emotion_detected]]+=1
            count_emotion_label[labelmap[data[1]]]+=1
    for j in range(len(emotion_labels)):
        print('accuracy of emotion : %s %f%%.' % (emotion_labels[j], count_emotion_result[j]/ count_emotion_label[j] * 100))
            
    print('Total accuracy: %f%%.' % (accuracy / total * 100))




# In[23]:

# # Create training and testing data
sen = c[3]
emo = c[0]
l = len(c[3])
limit = (9*l)//10
sente = c[2]
Data = create_data(sen[:limit], emo[:limit])
test_data = create_test(sente[limit:], emo[limit:]) 


 # In[26]:

 # get the training set and train the Naive Bayes Classifier
training_set = nltk.classify.util.apply_features(extract_features, Data)

classifier = NaiveBayes.train(training_set)
filename = 'final_model.sav'
pickle.dump(classifier,open(filename,'wb'))

load_model = pickle.load(open(filename,'rb')) 

# In[25]:

get_accuracy(test_data, load_model,emotion_labels )


