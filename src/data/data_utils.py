from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import pandas as pd
from glob import glob
import sys
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import codecs,string
import json
from torch.utils.data import Dataset
import argparse
import torch

def clean_tweets(text):
    text = text.lower()
    text = re.sub(r'@\w+','',text)
    text = re.sub(r'http\S+','',text)
    text = re.sub(r'://\S+','',text)
    text = re.sub(r'#\w+','',text)
    text = re.sub(r'\d+','',text)
    return text.strip()

def remove_html(text):
    text = text.replace("\n"," ")
    pattern = re.compile('<.*?>') #all the HTML tags
    return pattern.sub(r'', text)

def remove_email(text):
    text = re.sub(r'[\w.<>]*\w+@\w+[\w.<>]*', " ", text)
    return text

def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)
    
def language_identification(pipeline, text):
    results = pipeline(text)
    languages = {}
    text = ""
    
    for i, val in enumerate(results):
        if val['word'].startswith("##"):
            text += val['word'].replace("##",'')
        else:
            text += val['word']
        
        if i != len(results)-1:
            if results[i+1]['word'].startswith('##') == False:
                text += " "
                #languages.append(results[i]['entity'])
                languages[text.split()[-1]] = results[i]['entity']
        else:
            languages[text.split()[-1]] = results[i]['entity']
    
    return languages        

def calculate_CMI(line, lid_model):
    #tokenizer_ = AutoTokenizer.from_pretrained("sagorsarker/codeswitch-hineng-lid-lince")
    #model_ = AutoModelForTokenClassification.from_pretrained("sagorsarker/codeswitch-hineng-lid-lince")
    #lid_model = pipeline('ner', model=model_, tokenizer=tokenizer_)
    lids = language_identification(lid_model, line)
    total_count = 0
    hi_count = 0
    en_count = 0
    t = 0
    for token in list(lids.keys()):
        total_count += 1
        if lids[token] == 'hin':
            t = 1
        elif lids[token] == 'en':
            t = 2

        if t == 1:
            hi_count +=1
        elif t == 2:
            en_count += 1
    try:
        CMI = (total_count - max(hi_count, en_count))/total_count
    except:
        CMI = 0
    return CMI
    
def detect_language(character):
    maxchar = max(character)
    if u'\u0900' <= maxchar <= u'\u097f':
        return 'hindi'
    else:
        return 'english'
    
def get_year(x):
    l = re.findall(r'\d+ year',x)
    if len(l) > 0:
        return int(l[0].split(' ')[0])
    else:
        return 0