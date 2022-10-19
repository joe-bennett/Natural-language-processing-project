#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report
import prepare
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

# In[3]:

df=prepare.wrangle_data(pd.read_json('data.json'))
def make_word_list(df):
    """ creates a list of every not unique word in dataframe"""
    all_words = re.sub(r'[^\w\s]', '', (' '.join(df.lemmatized))).split()
    all_freq = pd.Series(all_words).value_counts()
    
    mask = all_freq > 1
    all_not_unique = list(all_freq[mask].index)
    
    return all_not_unique


# In[4]:


def finding_non_single_words(x):
    """finds all words in column that appear in df more than one time
    will be used to make a column that counts words that appear more than once"""
    all_not_unique = make_word_list(df)
    l = []
    for w in x:
        if w in all_not_unique:
            l.append(w)
    return l


# In[5]:


def feature_engineering(df):
    """creates calculated columns for df subsetted by type of column"""

    #list making features 
    df['word_list'] = df.lemmatized.apply(lambda x: re.sub(r'[^\w\s]', '', x).split())
    df['unique_words'] = df.word_list.apply(lambda x: pd.Series(x).unique())
    df['non_single_words'] = df.word_list.apply(lambda x: finding_non_single_words(x))

    # counting
    df['word_count_simple'] = df.lemmatized.str.count(" ") + 1
    df['word_count'] = df.word_list.apply(lambda x: len(x))
    df['unique_count'] = df.unique_words.apply(lambda x: len(x))
    df['non_single_count'] = df.non_single_words.apply(lambda x: len(x))

    # calculating
    df['percent_unique'] = (df.unique_count / df.word_count)
    df['percent_repeat'] = (1 - df.unique_count / df.word_count)
    df['percent_one_word'] = df.word_list.apply(lambda x: (pd.Series(x).value_counts() == 1).mean())
    df['percent_non_single'] = (df.non_single_count / df.word_count)

    return df


# In[ ]:

def freq_df(exp_df):
    """create all words and all freq for the freq_df"""
    all_words = (' '.join(exp_df.lemmatized))
    all_words = re.sub(r'[^\w\s]', '', all_words).split()
    all_freq = pd.Series(all_words).value_counts()

    js_words = (' '.join(exp_df[exp_df.language == 'JavaScript'].lemmatized))
    js_words = re.sub(r'[^\w\s]', '', js_words).split()
    js_freq = pd.Series(js_words).value_counts()

    tS_words = (' '.join(exp_df[exp_df.language == 'TypeScript'].lemmatized))
    tS_words = re.sub(r'[^\w\s]', '', tS_words).split()
    tS_freq = pd.Series(tS_words).value_counts()

    py_words = (' '.join(exp_df[exp_df.language == 'Python'].lemmatized))
    py_words = re.sub(r'[^\w\s]', '', py_words).split()
    py_freq = pd.Series(py_words).value_counts()

    go_words = (' '.join(exp_df[exp_df.language == 'Go'].lemmatized))
    go_words = re.sub(r'[^\w\s]', '', go_words).split()
    go_freq = pd.Series(go_words).value_counts()

    c_words = (' '.join(exp_df[exp_df.language == 'C++'].lemmatized))
    c_words = re.sub(r'[^\w\s]', '', c_words).split()
    c_freq = pd.Series(c_words).value_counts()

    freq_df = pd.DataFrame({'all': all_freq,
                       'JavaScript': js_freq,
                       'TypeScript': tS_freq,
                       'Python': py_freq,
                       'Go': go_freq,
                       'C++':c_freq
                        })
    freq_df = freq_df.fillna(0)
    freq_df = freq_df.astype(int)
    return freq_df

    
def bigram(exp_df):
    """create all word and all freq in prepare for the bigram"""
    all_words = (' '.join(exp_df.lemmatized))
    all_words = re.sub(r'[^\w\s]', '', all_words).split()
    all_freq = pd.Series(all_words).value_counts()

    js_words = (' '.join(exp_df[exp_df.language == 'JavaScript'].lemmatized))
    js_words = re.sub(r'[^\w\s]', '', js_words).split()
    js_freq = pd.Series(js_words).value_counts()

    tS_words = (' '.join(exp_df[exp_df.language == 'TypeScript'].lemmatized))
    tS_words = re.sub(r'[^\w\s]', '', tS_words).split()
    tS_freq = pd.Series(tS_words).value_counts()

    py_words = (' '.join(exp_df[exp_df.language == 'Python'].lemmatized))
    py_words = re.sub(r'[^\w\s]', '', py_words).split()
    py_freq = pd.Series(py_words).value_counts()

    go_words = (' '.join(exp_df[exp_df.language == 'Go'].lemmatized))
    go_words = re.sub(r'[^\w\s]', '', go_words).split()
    go_freq = pd.Series(go_words).value_counts()

    c_words = (' '.join(exp_df[exp_df.language == 'C++'].lemmatized))
    c_words = re.sub(r'[^\w\s]', '', c_words).split()
    c_freq = pd.Series(c_words).value_counts()
    """Returns dataframe of bigram counts for each language"""
    # twenty most frequent bigrams for all words
    top_20_all_bigrams = (pd.Series(nltk.ngrams(all_words, 2))
                    .value_counts()
                    .head(20))
    # twenty most frequent bigrams for javascript
    top_20_js_bigrams = (pd.Series(nltk.ngrams(js_words, 2))
                    .value_counts()
                    .head(20))
    # twenty most frequent bigrams for typescript
    top_20_tS_bigrams = (pd.Series(nltk.ngrams(tS_words, 2))
                    .value_counts()
                    .head(20))
    # twenty most frequent bigrams for python
    top_20_py_bigrams = (pd.Series(nltk.ngrams(py_words, 2))
                    .value_counts()
                    .head(20))
    # twenty most frequent bigrams for c++
    top_20_c_bigrams = (pd.Series(nltk.ngrams(c_words, 2))
                    .value_counts()
                    .head(20))
    # twenty most frequent bigrams for go
    top_20_go_bigrams = (pd.Series(nltk.ngrams(go_words, 2))
                .value_counts()
                .head(20))
    
    
    return (pd.concat([top_20_all_bigrams, top_20_js_bigrams, top_20_tS_bigrams,
                                     top_20_py_bigrams, top_20_c_bigrams,top_20_go_bigrams], axis=1, sort=True)
        .set_axis(['all_bigram','JavaScript', 'TypeScript', 'Python', 'C++','Go'], axis=1, inplace=False)
        .fillna(0)
        .apply(lambda s: s.astype(int)))


def bigram_clean(exp_df):
    """create all word and all freq in prepare for the bigram"""
    all_words = (' '.join(exp_df.clean_tokens))
    all_words = re.sub(r'[^\w\s]', '', all_words).split()
    all_freq = pd.Series(all_words).value_counts()

    js_words = (' '.join(exp_df[exp_df.language == 'JavaScript'].clean_tokens))
    js_words = re.sub(r'[^\w\s]', '', js_words).split()
    js_freq = pd.Series(js_words).value_counts()

    tS_words = (' '.join(exp_df[exp_df.language == 'TypeScript'].clean_tokens))
    tS_words = re.sub(r'[^\w\s]', '', tS_words).split()
    tS_freq = pd.Series(tS_words).value_counts()

    py_words = (' '.join(exp_df[exp_df.language == 'Python'].clean_tokens))
    py_words = re.sub(r'[^\w\s]', '', py_words).split()
    py_freq = pd.Series(py_words).value_counts()

    go_words = (' '.join(exp_df[exp_df.language == 'Go'].clean_tokens))
    go_words = re.sub(r'[^\w\s]', '', go_words).split()
    go_freq = pd.Series(go_words).value_counts()

    c_words = (' '.join(exp_df[exp_df.language == 'C++'].clean_tokens))
    c_words = re.sub(r'[^\w\s]', '', c_words).split()
    c_freq = pd.Series(c_words).value_counts()
    """Returns dataframe of bigram counts for each language"""
    # twenty most frequent bigrams for all words
    top_20_all_bigrams = (pd.Series(nltk.ngrams(all_words, 2))
                    .value_counts()
                    .head(20))
    # twenty most frequent bigrams for javascript
    top_20_js_bigrams = (pd.Series(nltk.ngrams(js_words, 2))
                    .value_counts()
                    .head(20))
    # twenty most frequent bigrams for typescript
    top_20_tS_bigrams = (pd.Series(nltk.ngrams(tS_words, 2))
                    .value_counts()
                    .head(20))
    # twenty most frequent bigrams for python
    top_20_py_bigrams = (pd.Series(nltk.ngrams(py_words, 2))
                    .value_counts()
                    .head(20))
    # twenty most frequent bigrams for c++
    top_20_c_bigrams = (pd.Series(nltk.ngrams(c_words, 2))
                    .value_counts()
                    .head(20))
    # twenty most frequent bigrams for go
    top_20_go_bigrams = (pd.Series(nltk.ngrams(go_words, 2))
                .value_counts()
                .head(20))
    
    
    return (pd.concat([top_20_all_bigrams, top_20_js_bigrams, top_20_tS_bigrams,
                                     top_20_py_bigrams, top_20_c_bigrams,top_20_go_bigrams], axis=1, sort=True)
        .set_axis(['all_bigram','JavaScript', 'TypeScript', 'Python', 'C++','Go'], axis=1, inplace=False)
        .fillna(0)
        .apply(lambda s: s.astype(int)))

def vis_cloud1(exp_df):
    '''create a visualized word cloud for all words'''

    all_words = (' '.join(exp_df.clean_tokens))
    all_words = re.sub(r'[^\w\s]', '', all_words).split()
    all_freq = pd.Series(all_words).value_counts()

    js_words = (' '.join(exp_df[exp_df.language == 'JavaScript'].clean_tokens))
    js_words = re.sub(r'[^\w\s]', '', js_words).split()
    js_freq = pd.Series(js_words).value_counts()

    tS_words = (' '.join(exp_df[exp_df.language == 'TypeScript'].clean_tokens))
    tS_words = re.sub(r'[^\w\s]', '', tS_words).split()
    tS_freq = pd.Series(tS_words).value_counts()

    py_words = (' '.join(exp_df[exp_df.language == 'Python'].clean_tokens))
    py_words = re.sub(r'[^\w\s]', '', py_words).split()
    py_freq = pd.Series(py_words).value_counts()

    go_words = (' '.join(exp_df[exp_df.language == 'Go'].clean_tokens))
    go_words = re.sub(r'[^\w\s]', '', go_words).split()
    go_freq = pd.Series(go_words).value_counts()

    c_words = (' '.join(exp_df[exp_df.language == 'C++'].clean_tokens))
    c_words = re.sub(r'[^\w\s]', '', c_words).split()
    c_freq = pd.Series(c_words).value_counts()
    
    all_cloud = WordCloud(background_color='white', height=1000, width=400).generate(' '.join(all_words))
    js_cloud = WordCloud(background_color='white', height=600, width=800).generate(' '.join(js_words))
    tS_cloud = WordCloud(background_color='white', height=600, width=800).generate(' '.join(tS_words))
    py_cloud = WordCloud(background_color='white', height=600, width=800).generate(' '.join(py_words))
    go_cloud = WordCloud(background_color='white', height=600, width=800).generate(' '.join(go_words))
    c_cloud = WordCloud(background_color='white', height=600, width=800).generate(' '.join(c_words))
    
    plt.figure(figsize=(10, 8))
    axs = [plt.axes([0, 0, .5, 1]), plt.axes([.5, .5, .5, .5]), plt.axes([.5, 0, .5, .5])]

    axs[0].imshow(all_cloud)
    axs[1].imshow(js_cloud)
    axs[2].imshow(tS_cloud)
#     axs[3].imshow(py_cloud)
#     axs[4].imshow(go_cloud)
#     axs[5].imshow(c_cloud)
    axs[0].set_title('All Words')
    axs[1].set_title('JavaScript')
    axs[2].set_title('TypeScript')
#     axs[3].set_title('Python')
#     axs[4].set_title('Go')
#     axs[5].set_title('C++')

    for ax in axs: ax.axis('off')

def vis_cloud2(exp_df):
    '''create a visualized word cloud for all words'''

    all_words = (' '.join(exp_df.clean_tokens))
    all_words = re.sub(r'[^\w\s]', '', all_words).split()
    all_freq = pd.Series(all_words).value_counts()

    js_words = (' '.join(exp_df[exp_df.language == 'JavaScript'].clean_tokens))
    js_words = re.sub(r'[^\w\s]', '', js_words).split()
    js_freq = pd.Series(js_words).value_counts()

    tS_words = (' '.join(exp_df[exp_df.language == 'TypeScript'].clean_tokens))
    tS_words = re.sub(r'[^\w\s]', '', tS_words).split()
    tS_freq = pd.Series(tS_words).value_counts()

    py_words = (' '.join(exp_df[exp_df.language == 'Python'].clean_tokens))
    py_words = re.sub(r'[^\w\s]', '', py_words).split()
    py_freq = pd.Series(py_words).value_counts()

    go_words = (' '.join(exp_df[exp_df.language == 'Go'].clean_tokens))
    go_words = re.sub(r'[^\w\s]', '', go_words).split()
    go_freq = pd.Series(go_words).value_counts()

    c_words = (' '.join(exp_df[exp_df.language == 'C++'].clean_tokens))
    c_words = re.sub(r'[^\w\s]', '', c_words).split()
    c_freq = pd.Series(c_words).value_counts()
    
    all_cloud = WordCloud(background_color='white', height=1000, width=400).generate(' '.join(all_words))
    js_cloud = WordCloud(background_color='white', height=600, width=800).generate(' '.join(js_words))
    tS_cloud = WordCloud(background_color='white', height=600, width=800).generate(' '.join(tS_words))
    py_cloud = WordCloud(background_color='white', height=600, width=800).generate(' '.join(py_words))
    go_cloud = WordCloud(background_color='white', height=600, width=800).generate(' '.join(go_words))
    c_cloud = WordCloud(background_color='white', height=600, width=800).generate(' '.join(c_words))
    
    plt.figure(figsize=(10, 8))
    axs = [plt.axes([0, 0, .5, 1]), plt.axes([.5, .5, .5, .5]), plt.axes([.5, 0, .5, .5])]


    axs[0].imshow(py_cloud)
    axs[1].imshow(go_cloud)
    axs[2].imshow(c_cloud)

    axs[0].set_title('Python')
    axs[1].set_title('Go')
    axs[2].set_title('C++')

    for ax in axs: ax.axis('off')

