
# coding: utf-8
"""
@author: dixit
"""


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
test_Data = pd.read_csv('test_table.csv',header=None)

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
    count = 0
    for sen in sentences:
#         print count
#         print sen
#         print type(sen)
        s = nltk.word_tokenize(sen)
        characters = ["á", "\xc3", "\xa1", "\n", ",", "."]
        new = ' '.join([i for i in s if not [e for e in characters if e in i]])
        sentence_list.append(new)
    return sentence_list


# In[7]:

'''
POS-TAGGER, returns NAVA words
'''
def pos_tag(sentences):
    tags = [] #have the pos tag included
    nava_sen = [] #have the sentence after pos tagging 
    for s in sentences:
        s_token = nltk.word_tokenize(s)
        pt = nltk.pos_tag(s_token)
        nava = []
        nava_words = []
        for t in pt:
            # words having length less than 3 removed
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
# sen_string store sentences after stemming
# sen_token stores list of tokens after stemming for each sentence        
# here senetence_list is     
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
    emotions = Data[0]#emotion of sentence from dataset
    sit = Data[1]#senetence from dataset
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


'''
Creating the dataframe
'''
def create_test_frame(Data):
    sentences = Data[0]
    print(sentences[1:])
    sent = removal(sentences[1:])
    nava, sent_pt = pos_tag(sent)
    sentences, sen_string, sen_token = stemming(sent_pt)
    frame = pd.DataFrame({0 : sen_string})
    return frame
# In[11]:

c = create_frame(Data)
t_frame = create_test_frame(test_Data)

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


# In[18]:

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
def create_test(sentence):    
    sen = []
    for s in sentence:
        sen.append(str(s))    
    return sen


# In[21]:

'''
Classifier
'''
def classify_dataset(data):
    return         classifier.classify(extract_features(nltk.word_tokenize(data)))


# In[22]:

'''
Get Emotion
'''
def get_Emotion(test_data, classifier):
    emotions_detected = []
    for data in test_data:
        emotion_detected = classify_dataset(data)
        emotions_detected.append(emotion_detected)        
    save_result(test_data, emotions_detected)
# In[23]
'''
save result in file
'''
def save_result(test_data, emotions_detected):
    raw_data = {'sentence': test_data, 
        'emotion': emotions_detected }
    df = pd.DataFrame(raw_data, columns = ['sentence', 'emotion'])
    print(df)
    #df.to_csv('result.csv')
# In[24]:

# # Create training and testing data
sen = c[3]
emo = c[0]
l = len(c[3])
limit = (9*l)//10
sente = c[2] #sentence string
Data = create_data(sen[:limit], emo[:limit])
test_data = create_test(t_frame[0]) 

 # In[25]:

 # extract the word features out from the training data
word_features = get_word_features(get_words_in_dataset(Data))
 # In[26]:

 # get the training set and train the Naive Bayes Classifier
 
training_set = nltk.classify.util.apply_features(extract_features, Data)

classifier = NaiveBayes.train(training_set)
filename = 'final_model.sav'
pickle.dump(classifier,open(filename,'wb'))

load_model = pickle.load(open(filename,'rb')) 

# In[25]:

get_Emotion(test_data, load_model)

