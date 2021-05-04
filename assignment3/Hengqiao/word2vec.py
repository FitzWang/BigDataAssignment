import os
import io
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV, RidgeClassifier, SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import text_classification

import torch
from torch import nn
from torch.nn import functional as F
import keras
from keras.layers import Embedding
from keras.models import Sequential
import gensim
from sklearn.manifold import TSNE

def document_vector_weighted(word2vec_model, doc,tfidf_dict):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in word2vec_model.vocab]
    
    weights = [tfidf_dict[word] for word in doc if word in word2vec_model.vocab]

    return np.average(word2vec_model[doc],axis=0,weights=weights)

def document_vector(word2vec_model, doc):
    # remove out-of-vocabulary words
    
    doc = [word for word in doc if word in word2vec_model.vocab]

    return np.mean(word2vec_model[doc], axis=0)

def has_vector_representation(word2vec_model, doc):
    """check if at least one word of the document is in the
    word2vec dictionary"""
    return not all(word not in word2vec_model.vocab for word in doc)

if __name__ == '__main__':
    setting_path = 'etc.json'
    with open(setting_path,"r") as load_f:
        setting = json.load(load_f)
    rejected_words = setting['rejected_words']
    not_related_words = setting['not_related_words']
    validation_portion = setting['validation_portion']
    min_df = setting['min_df']
    max_df = setting['max_df']
    ngram_range = (setting['ngram_range'][0],setting['ngram_range'][1])
    out_json_path = 'record_dict.json'

    if os.path.exists(out_json_path):
        with open(out_json_path,"r") as load_f:
            record_dict = json.load(load_f)
            print("loading...")
    else:
        record_dict = {"cv":{},"tfidf":{}}
        print ('creating...')

    df = pd.read_csv('df.csv',encoding='utf-8')
    df = text_classification.DropRep(df)
    
    df['tweet_text'] = text_classification.TransformMerge([text_classification.text_prepare(x,not_related_words,rejected_words) for x in df['tweet_text']],rejected_words)
    
    validation = df.sample(frac=validation_portion)
    train = df.drop(list(validation.index))
    X_train, y_train = train.tweet_text, train.label
    X_val, y_val = validation.tweet_text, validation.label
    
    
    print (df.head())
    print (df['label'].value_counts())


    tfidf = TfidfVectorizer()
    feature = tfidf.fit_transform(X_train)
    terms = tfidf.get_feature_names()
    sums = feature.sum(axis=0)

    tfidf_dict = {}
    for col, term in enumerate(terms):
        tfidf_dict[term] = sums[0,col]

    
    X_train = [i.split() for i in X_train]
    X_val = [i.split() for i in X_val]

    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

    weight_mode = 'tfidfweighted'

    def ToVecProcess(X_val,y_val,tfidf_dict,weighted):
        x =[]
        y = []
        del_idx = []
        for idx,doc in enumerate((X_val)): #look up each doc in model
            doc = [word for word in doc if word in word2vec_model.vocab]
            if len(doc) == 0:
                continue
            # try:
            if weighted == 'tfidfweighted':
                weights = [tfidf_dict[word] if word in tfidf_dict.keys() else 0.5 for word in doc]
                x.append(np.average(word2vec_model[doc],axis=0,weights=weights))
                y.append(y_val[idx])
            elif weighted == 'nonweighted':
                x.append(np.mean(word2vec_model[doc], axis=0))
                y.append(y_val[idx])
            # except:
            #     print (idx,doc)
            #     del_idx.append(idx)
        val_vec = np.array(x)
        return val_vec,y
    
    train_vec,y_train = ToVecProcess(list(X_train),list(y_train),tfidf_dict,weight_mode)
    val_vec,y_val = ToVecProcess(list(X_val),list(y_val),tfidf_dict,weight_mode)
    
    X_tsne = TSNE(n_components=2, verbose=2).fit_transform(train_vec)
    plt.figure(1, figsize=(30, 20),)

    label_dict = {"#vaccine":0,"#covid":1,"#china":2,"#biden":3,"#stopasianhate":4,"#inflation":5}
    y_train_c =[label_dict[i] for i in y_train]
    print (label_dict)

    plt.scatter(X_tsne[:, 0], X_tsne[:, 1],s=100, c=y_train_c, alpha=0.2)
    plt.savefig(weight_mode+"_tsne.png")
    # plt.show()
    record_mode_key = 'Word2Vec_'+weight_mode
    if record_mode_key not in record_dict.keys():
        record_dict[record_mode_key] = {}

    def print_evaluation_scores(y_val, predicted,predicted_prob,nlp_mode,learner_name,best_params_):
        accuracy=accuracy_score(y_val, predicted)
        recall_macro = recall_score(y_val, predicted,average='macro')
        recall_micro = recall_score(y_val, predicted,average='micro')
        f1_score_macro=f1_score(y_val, predicted, average='macro')
        f1_score_micro=f1_score(y_val, predicted, average='micro')
        f1_score_weighted=f1_score(y_val, predicted, average='weighted')
        macro_auc = roc_auc_score(y_val, predicted_prob,multi_class="ovo", average='macro')
        weighted_auc = roc_auc_score(y_val, predicted_prob,multi_class="ovo", average='weighted')
        record_dict[nlp_mode][learner_name] = {}
        record_dict[nlp_mode][learner_name]["best_params_"] = best_params_
        record_dict[nlp_mode][learner_name]["accuracy"] = accuracy
        record_dict[nlp_mode][learner_name]["recall_macro"] = recall_macro
        record_dict[nlp_mode][learner_name]["recall_micro"] = recall_micro
        record_dict[nlp_mode][learner_name]["f1_score_macro"] = f1_score_macro
        record_dict[nlp_mode][learner_name]["f1_score_micro"] = f1_score_micro
        record_dict[nlp_mode][learner_name]["f1_score_weighted"] = f1_score_weighted
        record_dict[nlp_mode][learner_name]["macro_auc"] = macro_auc
        record_dict[nlp_mode][learner_name]["weighted_auc"] = weighted_auc
        print("accuracy:",accuracy)
        print("recall_macro:",recall_macro)
        print("recall_micro:",recall_micro)
        print("f1_score_macro:",f1_score_macro)
        print("f1_score_micro:",f1_score_micro)
        print("f1_score_weighted:",f1_score_weighted)
        print("macro_auc:",macro_auc)
        print("weighted_auc:",weighted_auc)

    def TrainProcess(nlp_mode,learner_name,learner,train_vec,y_train,val_vec,y_val):
        
        learner.fit(train_vec,y_train)
        predicted = learner.predict(val_vec)
        if learner_name == 'SGD':
            calibrator = CalibratedClassifierCV(learner.fit(train_vec,y_train), cv='prefit')
            model=calibrator.fit(train_vec,y_train)
            predicted_prob = model.predict_proba(np.array(val_vec))
        else:
            predicted_prob = learner.predict_proba(np.array(val_vec))
        print (nlp_mode,learner)
        print (predicted_prob[0])
        print_evaluation_scores(y_val,predicted,predicted_prob,nlp_mode,learner_name,{})

    learner = LogisticRegression(multi_class='ovr',max_iter=1000)
    TrainProcess(record_mode_key,'LR',learner,train_vec,y_train,val_vec,y_val)

    learner = SGDClassifier()
    TrainProcess(record_mode_key,'SGD',learner,train_vec,y_train,val_vec,y_val)

    learner = RandomForestClassifier()
    TrainProcess(record_mode_key,'RF',learner,train_vec,y_train,val_vec,y_val)

    out_json_dict = json.dumps(record_dict,indent=4)
    f3 = open(out_json_path, 'w')
    f3.write(out_json_dict)
    f3.close()

    