#!/usr/bin/env python
# coding: utf-8

# In[23]:


import spacy_dbpedia_spotlight
import spacy
import pandas as pd
import time
import requests
from difflib import SequenceMatcher
from tqdm import tqdm
import csv
import numpy as np


# In[2]:


#Read TextCorpus for Annotation
df = pd.read_pickle('data/textCorpusCleanedV0.2')
df.shape


# In[3]:


spotlight = spacy_dbpedia_spotlight.load('en')
sentencizer = spacy.load('en_core_web_sm')


# In[4]:


#Split sentences to rows to be able to distinguish sentences with multiple or no entities.
def split_sents(text):
    entities =[]
    #Due to long texts, we gotta split them for the requests..
    doc = sentencizer(text)
    sentences = [sent.string.strip() for sent in doc.sents]
    return sentences

df["text"] = df["text"].apply(lambda row: split_sents(row) )
df = df.explode("text")
df.reset_index(inplace=True)


# In[5]:


#Annotation method
def annotate_text(text):
    entities =[]
    #Due to long texts, we gotta split them for the requests..
    #doc = sentencizer(text)
    #sentences = [sent.string.strip() for sent in doc.sents]
    
    #for sentence in sentences:
    try:
        doc2 = spotlight(text)
        entities.extend([(ent.text, ent.kb_id_) for ent in doc2.ents])
    except Exception as e:
        print('Error ', e, ' at ', text) #sentence)
    #entities =  [(ent.text, ent.label_, ent.kb_id_) for ent in doc.ents]
    
    
    return list(dict.fromkeys(entities)) #Return only unique entries..


# In[6]:


#%%time
df['entities'] = df['text'].apply(lambda text: annotate_text(text))


# In[7]:


#Get a list of all entities contained
all_entity_list = df.entities.sum()
#Remove duplicates
unique_entities = list(dict.fromkeys(all_entity_list))

print(len(unique_entities), ' entities need to be matched')


# In[8]:


#Test Setting with only 10 entities
#test1 = unique_entities[:10]


# In[9]:


#%%time

illegalDrugIndicators = ['dbp:legalAu', 'dbp:legalCa', 'dbp:legalDe', 'dbp:legalNz', 
                         'dbp:legalUk', 'dbp:legalUn', 'dbp:legalUs']
list_of_drugs = []
url = 'http://dbpedia.org/sparql'

#for ent_tuple in unique_entities:
for ent_tuple in tqdm(unique_entities):
    
    for ind in illegalDrugIndicators:
        query = """
                select * where {
                <""" + ent_tuple[1] + "> " + ind + " " + "?c" + """ }
                LIMIT 100
                """
        try:
            r = requests.get(url, params = {'format': 'json', 'query': query})
            data = r.json()

            if len(data['results']['bindings']) > 0:
                list_of_drugs.append(ent_tuple)
                break
        except Exception as e:
            print('Error ', e, ' at ', ent_tuple, ' and relation ', ind)
        
        


# In[10]:


#The resource link is unique and needs to be the identifier. Otherwise different writings linked to the known drug might be ignored..
drug_dict = {y:x for x,y in dict(list_of_drugs).items()}
drug_dict


# In[11]:


def label_drugs_in_text(entity_list, text, lookup_dict):
    final_text = text
    for candidate in entity_list:
        if candidate[1] in lookup_dict:
            final_text = final_text.replace(candidate[0], '<DRUG> '+candidate[0]+ ' </DRUG>')
    return final_text

def create_drug_label_list(entity_list, text, lookup_dict):
    labels = []
    for candidate in entity_list:
        if candidate[1] in lookup_dict:
            labels.append((candidate[0], 'DRUG'))
    return labels
            


# In[12]:


#df["newText"] = df.apply(lambda row: create_drug_label_list(row['entities'], row['text'], drug_dict2), axis=1)
df["labels"] = df.apply(lambda row: create_drug_label_list(row['entities'], row['text'], drug_dict), axis=1)


# In[13]:


#Store raw
df.to_pickle('data/raw_annotated_DF_V0.2')


# In[14]:


#Remove sentences without Drug label
index_to_del = df[ df['labels'].map(lambda d: len(d)) <= 0 ].index
df.drop(index_to_del, inplace=True)


# In[24]:


#Store reduced Sentence set as Domain Text corpora
np.savetxt('data/reducedTextCorpusV0.2.txt', df["text"].values, fmt = "%s")


# In[15]:


#Convert to BIO-NER task format

def matcher(string, pattern):
    '''
    Return the start and end index of any pattern present in the text.
    '''
    match_list = []
    pattern = pattern.strip()
    seqMatch = SequenceMatcher(None, string, pattern, autojunk=False)
    match = seqMatch.find_longest_match(0, len(string), 0, len(pattern))
    if (match.size == len(pattern)):
        start = match.a
        end = match.a + match.size
        match_tup = (start, end)
        string = string.replace(pattern, "X" * len(pattern), 1)
        match_list.append(match_tup)
        
    return match_list, string

def mark_sentence(s, match_list):
    '''
    Marks all the entities in the sentence as per the BIO scheme. 
    '''
    word_dict = {}
    for word in s.split():
        word_dict[word] = 'O'
        
    for start, end, e_type in match_list:
        temp_str = s[start:end]
        tmp_list = temp_str.split()
        if len(tmp_list) > 1:
            word_dict[tmp_list[0]] = 'B-' + e_type
            for w in tmp_list[1:]:
                word_dict[w] = 'I-' + e_type
        else:
            word_dict[temp_str] = 'B-' + e_type
    return word_dict

''' NOT NECESSARY SINCE CLEANING IS DONE ELSEWHERE
def clean(text):
    
    #Just a helper fuction to add a space before the punctuations for better tokenization
    filters = ["!", "#", "$", "%", "&", "(", ")", "/", "*", ".", ":", ";", "<", "=", ">", "?", "@", "[",
               "\\", "]", "_", "`", "{", "}", "~", "'"]
    for i in text:
        if i in filters:
            text = text.replace(i, " " + i)
            
    return text
'''

def create_data(df, filepath):
    '''
    The function responsible for the creation of data in the said format.
    '''
    with open(filepath , 'w') as f:
        for text, annotation in zip(df.text, df.annotation):
            #text = clean(text) # Cleaning need to be done in preprocessing
            text_ = text        
            match_list = []
            for i in annotation:
                a, text_ = matcher(text, i[0])
                match_list.append((a[0][0], a[0][1], i[1]))

            d = mark_sentence(text, match_list)

            for i in d.keys():
                f.writelines(i + ' ' + d[i] +'\n')
            f.writelines('\n')
            


# In[16]:


## Subset and rename Train
bio_df = df[["text", "labels"]].copy()
bio_df.rename(columns = {'text':'text', 'labels':'annotation'}, inplace = True)


# In[17]:


#Create and Store NER task file
train_filepath = 'data/bio_drugs_corpusV0.2.txt'
create_data(bio_df, train_filepath)


# In[ ]:





# In[ ]:




