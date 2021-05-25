import streamlit as st

import torch
import torch.nn as nn
import os
from transformers import AutoConfig
from transformers import AutoModelWithLMHead
from transformers import AutoTokenizer
from src import model
import pickle
import numpy as np
import spacy
from spacy import displacy

st.title('DRUG Detector')


st.write('Please insert text into textbox and press run once you want to detect entities')

drug_ad = st.text_input('Drug Text as input for NER', value='E.g. Selling the best meth in EU!',
                        max_chars=None, key=None, type='default')

#HTML wrapper for displacy
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

#Load Parameters
model_path = "experiments/drugs/6/state_dict.pth"
params_path = "experiments/drugs/6/params.pkl"
with open(params_path, 'rb') as f:
    params = pickle.load(f)

@st.cache
def load_model():
    #Initialize Model from CrossNER
    tagger = model.BertTagger(params)

    #Load trained model parameters
    state_dict = torch.load(model_path)
    tagger.load_state_dict(state_dict)
    return tagger

@st.cache
def preprocess_Input(input_text):
    auto_tokenizer = AutoTokenizer.from_pretrained(params.model_name)
    test_text_arr = input_text.split()
    subs_ = [auto_tokenizer.tokenize(token) for token in test_text_arr]

    token_to_subs = {}
    tokens = []
    sid = 0
    for i, sub_list in enumerate(subs_):
        tokens.extend(auto_tokenizer.convert_tokens_to_ids(sub_list))
        token_to_subs[i] = []
        for sub in sub_list:
            token_to_subs[i].append(sid)
            sid += 1

    tokens = [auto_tokenizer.cls_token_id] + tokens + [auto_tokenizer.sep_token_id]
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0), token_to_subs

@st.cache
def predict_text(tagger, tokens):
    tagger.eval()
    preds = tagger(tokens)
    preds = preds.detach().cpu().numpy()
    preds = np.concatenate(preds, axis=0)
    return np.argmax(preds, axis=1)

@st.cache
def convert_pred_to_spacy_format(predictions, token_to_sub_ids, orig_text):
    # Remove CLS and SEP token
    pred_subs = predictions[1:len(predictions) - 1]
    tokens = orig_text.split()

    final_text = ""
    ent_list = []

    for t_id, token in enumerate(tokens):
        drug = False
        # Check each token if it has at least one drug prediction
        for sid in token_to_sub_ids[t_id]:
            if pred_subs[sid] == 9 or pred_subs[sid] == 10:
                drug = True
            elif pred_subs[sid] > 0:
                raise Exception('Attention there is another prediction! Adjust algorithm')

        if drug:
            ent_list.append({'end': len(final_text) + len(token), 'label': 'DRUG', 'start': len(final_text)})
        # since we probably have multiple whitespaces I wanna be on the safe side and rebuild the text
        final_text = final_text + token + " "

    return {'ents': ent_list, 'text': final_text}

if st.button('Predict input text'):
    if len(drug_ad) > 20:

        #Load model and predict sentence (all cached)
        ner_tagger = load_model()
        token_list, token_to_subs = preprocess_Input(drug_ad)
        predictions = predict_text(ner_tagger, token_list)

        #convert to spacy format and display Displacy Visualization
        spacy_predictions = convert_pred_to_spacy_format(predictions, token_to_subs, drug_ad)
        
        html = displacy.render(spacy_predictions, style='ent', manual=True)
        # Newlines seem to mess with the rendering
        html = html.replace("\n", " ")
        
        st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
    else:
        st.write('Please enter a real text with at least 20 chars..')









