
# coding: utf-8

# In[2]:

import pandas as pd
import collections
import matplotlib.pyplot as plt
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
import re
import collections
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
pd.set_option('display.max_column',4000)
pd.set_option('display.max_row',4000)


# # 1. Reading Training and Test data into dataframe

# In[3]:

tr=pd.read_csv("P3_training_data.csv",header=None)
te=pd.read_csv("P3_test_data.csv",header=None)
print(te)
tr.columns=['text']
tr=tr['text'].str.split(n=1,expand=True)
tr.columns=['class','text']
te.columns=['text']
te['class']='none'
print(tr)
print(te)


# # 2. DEFINING FUNCTIONS FOR STOPWORD REMOVAL,STEMMIZATION AND           CLEANING PUNCTUATIONS
# 

# In[4]:

def cleanpunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return  cleaned
stop = set(stopwords.words('english'))
print(stop)
sno = nltk.stem.SnowballStemmer('english')
def preprocessing(x):
    final_string=[]
    s=''
    for text in x['text'].values:
        filtered_sentence=[]
        for w in text.split():
            for cleaned_words in cleanpunc(w).split():
                if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):    
                    if(cleaned_words.lower() not in stop):
                        s=(sno.stem(cleaned_words.lower())).encode('utf8')
                        filtered_sentence.append(s)
               
                    else:
                        continue
                else:
                    continue 
        str1 = b" ".join(filtered_sentence)
        str1=str1.decode('utf8')
        final_string.append(str1)

    x['text']=final_string


# # 3. Apply Preprocessing to both training and test data

# In[5]:

preprocessing(tr)
preprocessing(te)
print(tr)


# # 4. Seperating training data to positive and negative class

# In[6]:

tr_pos=tr[tr['class']=='Pos']
tr_neg=tr[tr['class']=='Neg']


# In[7]:

tr_pos_count=tr_pos.shape[0]
tr_neg_count=tr_neg.shape[0]
prior_pos=tr_pos_count/(tr_pos_count+tr_neg_count)
prior_neg=1-prior_pos


# # 5. Creating multinomial BOW representations for both positive and negative class words

# In[8]:

pos_voc=dict(collections.Counter([y for x in tr_pos.text.values.flatten() for y in x.split()]))
neg_voc=dict(collections.Counter([y for x in tr_neg.text.values.flatten() for y in x.split()]))
print(neg_voc)


# In[9]:

voc=set(list(pos_voc.keys())+list(neg_voc.keys()))
voc_size=np.size(list(voc))
print(voc_size)


# In[10]:

pos_voc_count=0
for value in pos_voc.values():
    pos_voc_count=pos_voc_count+value
neg_voc_count=0
for value in neg_voc.values():
    neg_voc_count=neg_voc_count+value


# # 6. Implementing Bayes Classifier for predicted the class for test data

# In[11]:

def pos_prob(x):
    if x in pos_voc.keys():
        x_count=pos_voc[x]+1
    elif (x in neg_voc.keys()):
        x_count=1
    else:
        return 0
    
    return np.log((x_count/(pos_voc_count+1*voc_size)))
def neg_prob(x):
    if x in neg_voc.keys():
        x_count=neg_voc[x]+1
    elif (x in pos_voc.keys()):
        x_count=1
    else:
        return 0
    
    return np.log((x_count/(neg_voc_count+1*voc_size)))


# In[12]:

output=[]

for text in te['text']:
    
    l=text.split()
    pos_prob_total=0
    neg_prob_total=0
    for i in l:
        pos_prob_total=pos_prob_total+pos_prob(i)
        neg_prob_total=neg_prob_total+neg_prob(i)
    if(pos_prob_total>neg_prob_total):
        output=output+[1]
    else:
        output=output+[0]
print(len(output))
print(output)


# In[ ]:




# In[ ]:




# In[ ]:



