#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd

import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords


# In[33]:


def basic_clean(string):
    string = string.lower()
    string = (unicodedata.normalize('NFKD', string)
                         .encode('ascii', 'ignore')
                         .decode('utf-8', 'ignore')
             )
    string = re.sub(r"[^a-z0-9'\s]", '', string)
    return string


# In[34]:


def clean_html(string):
    string = re.sub(r'<[^>]*>', '', string)
    string = re.sub(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)", '', string)
    string = re.sub(r'\n', '', string)
    string = re.sub(r'\s\s', '', string)
    return string


# In[35]:


def tokenize(string):
    tokenizer = nltk.tokenize.ToktokTokenizer()
    return tokenizer.tokenize(string, return_str=True)


# In[36]:


def stem(string):
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in string.split()]
    return ' '.join(stems)


# In[45]:


def lemma(string):
    wnl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    return ' '.join(lemmas)


# In[38]:


def remove_stopwords(string, extra_words=[], exclude_words=[]):
    stopword_list = stopwords.words('english')
    
    for word in extra_words:
        stopword_list.append(word)
    
    for word in exclude_words:
        stopword_list.remove(word)
        
    words = string.split()
    filtered_words = [word for word in words if word not in stopword_list]
    return ' '.join(filtered_words)


# In[46]:


def prepare_readme_data(df,column):
    clean_tokens = (df[column].apply(clean_html)
                              .apply(basic_clean)
                              .apply(tokenize)
                              .apply(remove_stopwords)
                   )
    
    for token in clean_tokens:
        token = ' '.join(token).split()
    
    df['stemmed'] = clean_tokens.apply(stem)
    df['lemmatized'] = clean_tokens.apply(lemma)
    df['clean_tokens']=clean_tokens
    return df


# In[51]:


def wrangle_data(target):
    data = target
    return prepare_readme_data(data, 'readme_contents')


# In[50]:





# In[31]:





# In[ ]:




