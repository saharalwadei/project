#!/usr/bin/env python
# coding: utf-8

# Installing required libraries

# In[ ]:


import sys
get_ipython().system('{sys.executable} -m pip install pandas')
get_ipython().system('{sys.executable} -m pip install numpy')
get_ipython().system('{sys.executable} -m pip install nltk')
get_ipython().system('{sys.executable} -m pip install emoji --upgrade')


# Importing required libraries

# In[3]:


import pandas as pd
import numpy as np 

import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem.isri import ISRIStemmer

import csv
import re # for regular expression
import string
import emoji


# In[4]:


#get the tweets text only 
raw_tweets = []
txt_files = [#'data/raw/tweets_الاتصالات_السعودية.txt', \
             #'data/raw/tweets_STC.txt'] #,
             'data/raw/tweets_STCcare.txt'] 
             #'data/raw/tweets_STC_KSA.txt'
for file in txt_files:
    file = open(file, "r")
    for line in file:
        try:
            raw_tweets.append(line.split(';')[4])
        except:
            continue 
'''
    with open("raw_tweets.txt", "w") as txtFile:
        for tweet in raw_tweets:
            #print(tweet)
            txtFile.write(tweet)
'''
with open("raw_tweets.csv", "w") as csvFile:
    writer = csv.writer(csvFile, lineterminator='\n')
    for tweet in raw_tweets:
        writer.writerow([tweet])  


# In[5]:


df = pd.read_csv('raw_tweets.csv', header=None, names=['tweet'])#,'label' needed later
df.head(20)


# ##### Pre-proccssing Functions

# In[6]:


#An idea of removing noise data by using lexicon, I belive this should be impleneted earlier than labeling stage

# Remove nosiy tweets
noise=["أرباح", "للتنازل" , "سيولة", "نتائج","الوظائف" ,"وظائف"] #,"","", "","", ""
def remove_noise(tweet):
    label="ok"
    for word in noise:
        if word in tweet:
            label="noise"

    return label

# apply the method
#df["is Noise"] = df['tweet'].apply(lambda x: remove_noise(x))

# remove the noise
#df = df[df["is Noise"]!="noise"]


# In[7]:


def remove_stopwords(text):
    temp_text = ""
    for word in text.split(): 
        if not word in set(stopwords.words('arabic')):
            temp_text += word + " "
    text = temp_text
    return text


# In[8]:


def replace_bad_words(text):
    offensive = ['لعن','اللعنة', 'ملعونه','ملعونة', 'عيال', 'الكلب','ولعن'
                 ,'وملعونه','وملعونة','مقرف','مخيس','مخيسة',
                 'المخيسة','الملعنة','الخايسة','زبالة','زباله',
                 'زفت','الزفت','زق','','ازق','مسخرة','المسخرة','الخايس']
    temp_text = ""
    for word in text.split():
        if word in offensive:
            temp_text += 'offensive' + " "
        else:
            temp_text += word + " "
    text = temp_text
    return text


# In[9]:


def remove_zero_impact_words(text):
    delete = ['الاتصالات', 'السعودية', 'الاتصالاتالسعودية','الاتصالاتالسعوديه','السعوديه','السلام','عليكم','شركة','شركات','شركه','خدمه','خدمة','الإتصالات', 'الإتصالاتالسعودية','الإتصالاتالسعوديه']
    delete.append(['شركه','و','خدمه','الي','اله','ال','عليكم','انا','واله' ,'ب','السلام'])
    temp_text = ""
    for word in text.split():
        if word not in delete:
            temp_text += word + " "
    text = temp_text
    return text


# In[10]:


def stem(text): #[st.stem(word) for word in text if not word in set(stopwords.words('english'))]
    st = ISRIStemmer()
    temp_text = ""
    for word in text.split(): 
        #print(st.stem(word))
        temp_text += st.stem(word) + " "
    text = temp_text
    return text


# In[11]:


def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    return text


# In[12]:


def remove_diacritics(text):
    diacritics = re.compile(""" ّ   | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
    text = re.sub(diacritics, '', text)
    return text


# In[13]:


# remove repeated letters
def remove_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)


# In[14]:


def remove_Eng_Char(text):
    
    #Regex @[A-Za-z0-9]+ represents mentions and #[A-Za-z0-9]+ represents English hashtags 
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)", " ", text).split())

    #Regex \w+:\/\/\S+ matches all the URLs starting with http:// or https:// and replacing it with space.
    text = ' '.join(re.sub("(\w+:\/\/\S+)", " ", text).split())

    #remove english letters
    text= ' '.join(re.sub('[a-z]+'," ", text).split())
    text= ' '.join(re.sub('[A-Z]+'," ", text).split())
    
    #remove numbers
    text = ''.join(i for i in text if not i.isdigit())
    
    return text


# In[15]:


# Replace emojis or emoticons to words
# emoji package will result in english words 
# for emoticons, defined dictionary will be used
# the dictionary could be farther extended by many emoticons and different styles of emoticons, e.g. Japanese

emoticons = {
        ":‑)":"smiley",
        ":-]":"smiley",
        ":-3":"smiley",
        ":->":"smiley",
        "8-)":"smiley",
        ":-}":"smiley",
        ":)":"smiley",
        ":]":"smiley",
        ":3":"smiley",
        ":>":"smiley",
        "8)":"smiley",
        ":}":"smiley",
        ":o)":"smiley",
        ":c)":"smiley",
        ":^)":"smiley",
        "=]":"smiley",
        "=)":"smiley",
        ":-))":"smiley",
        ":‑D":"smiley",
        "8‑D":"smiley",
        "x‑D":"smiley",
        "X‑D":"smiley",
        ":D":"smiley",
        "8D":"smiley",
        "xD":"smiley",
        "XD":"smiley",
        ":‑(":"sad",
        ":‑c":"sad",
        ":‑<":"sad",
        ":‑[":"sad",
        ":(":"sad",
        ":c":"sad",
        ":<":"sad",
        ":[":"sad",
        ":-||":"sad",
        ">:[":"sad",
        ":{":"sad",
        ":@":"sad",
        ">:(":"sad",
        ":'‑(":"sad",
        ":'(":"sad",
        ":‑P":"playful",
        "X‑P":"playful",
        "x‑p":"playful",
        ":‑p":"playful",
        ":‑Þ":"playful",
        ":‑þ":"playful",
        ":‑b":"playful",
        ":P":"playful",
        "XP":"playful",
        "xp":"playful",
        ":p":"playful",
        ":Þ":"playful",
        ":þ":"playful",
        ":b":"playful",
        "<3":"love"}
def replace_emojis_emoticons(text):
    words = text.split()
    reformed = [emoticons[word] if word in emoticons else word for word in words]
    text = " ".join(reformed)
    text = emoji.demojize(text)
    text = text.replace(":"," ")
    text = text.replace("_","")
    text = ' '.join(text.split())
    return text


# In[16]:


# Remove emojis or emoticons to words
# emoji package will result in english words 
# for emoticons, defined dictionary will be used
# the dictionary could be farther extended by many emoticons and different styles of emoticons, e.g. Japanese

emoticons = {
        ":‑)":"smiley",
        ":-]":"smiley",
        ":-3":"smiley",
        ":->":"smiley",
        "8-)":"smiley",
        ":-}":"smiley",
        ":)":"smiley",
        ":]":"smiley",
        ":3":"smiley",
        ":>":"smiley",
        "8)":"smiley",
        ":}":"smiley",
        ":o)":"smiley",
        ":c)":"smiley",
        ":^)":"smiley",
        "=]":"smiley",
        "=)":"smiley",
        ":-))":"smiley",
        ":‑D":"smiley",
        "8‑D":"smiley",
        "x‑D":"smiley",
        "X‑D":"smiley",
        ":D":"smiley",
        "8D":"smiley",
        "xD":"smiley",
        "XD":"smiley",
        ":‑(":"sad",
        ":‑c":"sad",
        ":‑<":"sad",
        ":‑[":"sad",
        ":(":"sad",
        ":c":"sad",
        ":<":"sad",
        ":[":"sad",
        ":-||":"sad",
        ">:[":"sad",
        ":{":"sad",
        ":@":"sad",
        ">:(":"sad",
        ":'‑(":"sad",
        ":'(":"sad",
        ":‑P":"playful",
        "X‑P":"playful",
        "x‑p":"playful",
        ":‑p":"playful",
        ":‑Þ":"playful",
        ":‑þ":"playful",
        ":‑b":"playful",
        ":P":"playful",
        "XP":"playful",
        "xp":"playful",
        ":p":"playful",
        ":Þ":"playful",
        ":þ":"playful",
        ":b":"playful",
        "<3":"love"}
def remove_emojis_emoticons(text):
    words = text.split()
    reformed = [" " if word in emoticons else word for word in words]
    text = " ".join(reformed)
    text = emoji.demojize(text)
    text = ' '.join(re.sub("(\:[A-Za-z0-9_]+)", " ", text).split())
    return text


# In[17]:


# remove punctuations

def remove_punctuations(text):
    #text = ' '.join(re.sub("[\/\.\,\!\?\:\;\-\_\=؛\،\؟]", " ", text).split())
    text = ' '.join(re.sub("[\…\*\”\"\(\)\«\»\/\.\,\!\?\:\;\-\_\=؛\،\؟]|'...'", " ", text).split())
    return text


# In[18]:


def remove_hashtags(text):
    text = ' '.join(re.sub('#([^\s]+)', r'\1', text).split())
    #text = ' '.join(re.sub((r"(?:\@|https?\://)\S+"), text).split())
    return text


# In[19]:


def preprocess_tweet(text):

    # Remove all the hashtags as hashtags do not affect sentiments in this case
    # Replace #word with word 
    #text= remove_hashtags(text)
    
    # remove English characters as the tweets are in arabic and this also include usernames mentions and URLs 
    text= remove_Eng_Char(text)
    
    # remove diacritics from each word ot the text as it has alomost no impact on sentiments written in dialogue lang
    text= remove_diacritics(text)
    
    # replace emojis and emoticons as they has a great impact on sentiments
    # note that should be done after removing any other undesired characters such as characters in English, usernames, hashtags, ...etc.
    # this is also should be performed before removing any punctiuation marks as people use them to express their emotions occasionally 
    text= replace_emojis_emoticons(text)
    #text= remove_emojis_emoticons(text)
    
    # remove punctuations after convert emojis and emoticons to words
    text= remove_punctuations(text)
    
    # remove remove zero impact words 
    text= replace_bad_words(text)
    
    # remove stopwords 
    text= remove_stopwords(text)
    
    # normalize the tweet
    text= normalize_arabic(text)
    
    # remove repeating charachters as they are common in dialogue lang
    text= remove_repeating_char(text)
    
    # remove remove zero impact words 
    #text= remove_zero_impact_words(text) #repeat it file as _4
    
    # stem the text 
    #text= stem(text)
    
    return text


# In[20]:


# preprocess the tweets text and creat another cloumn with the processed text
df['clean_tweet'] = df['tweet'].apply(lambda x: preprocess_tweet(x))


# In[21]:


new_df = df[['clean_tweet']]
export_csv = new_df.to_csv ('cleaned_tweets.csv', index = None, header=True) 


# In[22]:


new_df.head()


# In[ ]:




