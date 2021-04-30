
import json
import os
from ast import literal_eval
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse as sp_sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import naive_bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import nltk
import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import gensim
from gensim.models.doc2vec import Doc2Vec
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk import pos_tag


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
#     elif treebank_tag.startswith('R'):
#         return wordnet.ADV
    else:
        return ''

def TransformMerge(texts,rejected_words):
    lmtzr = WordNetLemmatizer()
    
    words_dict = {}
    for idx_,text in enumerate(texts):
        words = text.split()
        new_words = []
        for word in words:
            # if word in rejected_words:
            #     continue
            if word  in words_dict:
                new_words.append(words_dict[word])
                continue
                
            tag = pos_tag(word_tokenize(word)) # tag is like [('bigger', 'JJR')]

            pos = get_wordnet_pos(tag[0][1])
            if pos:
                lemmatized_word = lmtzr.lemmatize(word, pos)
    #                 print ([tag,pos,lemmatized_word])
                if lemmatized_word in rejected_words:
                    continue
                words_dict[word] = lemmatized_word
                new_words.append(words_dict[word])
            else:
                new_words.append(word)
        texts[idx_] = ' '.join(new_words)
                

    return texts


def replace_abbreviations(text):
    # patterns that used to find or/and replace particular chars or words
    
    new_text = text
    
    # to find chars that are not a letter, a blank or a quotation
    pat_letter = re.compile(r'[^a-zA-Z \']+')
    new_text = pat_letter.sub(' ', text).strip().lower()
        
    # to find the 's following the pronouns. re.I is refers to ignore case
    pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)
    new_text = pat_is.sub(r"\1 is", new_text)
    
    # to find the 's following the letters
    pat_s = re.compile("(?<=[a-zA-Z])\'s")
    new_text = pat_s.sub("", new_text)
    
    # to find the ' following the words ending by s
    pat_s2 = re.compile("(?<=s)\'s?")
    new_text = pat_s2.sub("", new_text)
    
    # to find the abbreviation of not
    pat_not = re.compile("(?<=[a-zA-Z])n\'t")
    new_text = pat_not.sub(" not", new_text)
    
    # to find the abbreviation of would
    pat_would = re.compile("(?<=[a-zA-Z])\'d")
    new_text = pat_would.sub(" would", new_text)
    
    # to find the abbreviation of will
    pat_will = re.compile("(?<=[a-zA-Z])\'ll")
    new_text = pat_will.sub(" will", new_text)
    
    # to find the abbreviation of am
    pat_am = re.compile("(?<=[I|i])\'m")
    new_text = pat_am.sub(" am", new_text)
    
    # to find the abbreviation of are
    pat_are = re.compile("(?<=[a-zA-Z])\'re")
    new_text = pat_are.sub(" are", new_text)
    
    # to find the abbreviation of have
    pat_ve = re.compile("(?<=[a-zA-Z])\'ve")
    new_text = pat_ve.sub(" have", new_text)
    
    new_text = new_text.replace('\'', ' ')
    
    return new_text

def Doc2VecModel():
    model = gensim.models.Doc2Vec(size=100, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025)

    train = [text_prepare(x,not_related_words,rejected_words).split(' ') for x in train_df.tweet_text]
    train = [gensim.models.doc2vec.TaggedDocument(train[i],list(train_df['label'])[i]) for i in range(len(train))]
    model.build_vocab(train)
    for epoch in range(10):
        model.train(train,total_examples=model.corpus_count, epochs=model.iter)
        model.alpha -= 0.002


    doc_train=[]
    for idx,sub_X_train in enumerate(X_train):
        invec1 = model.infer_vector(sub_X_train.split(' '), alpha=0.1, min_alpha=0.0001, steps=5)
        doc_train.append(invec1)



    doc_test=[]
    for idx,sub_X_val in enumerate(X_val):
        invec1 = model.infer_vector(sub_X_val.split(' '), alpha=0.1, min_alpha=0.0001, steps=5)
        doc_test.append(invec1)
    
    def SubTrainProcess(learner_name,learner):
        
        learner.fit(doc_train,m_label_train)
        predicted=learner.predict(np.array(doc_test))
        if learner_name == 'SGD':
            calibrator = CalibratedClassifierCV(learner.fit(doc_train,m_label_train), cv='prefit')
            model=calibrator.fit(doc_train,m_label_train)
            predicted_prob = model.predict_proba(np.array(doc_test))
        else:
            predicted_prob = learner.predict_proba(np.array(doc_test))
        print (predicted_prob[:10])
        print_evaluation_scores(y_val, predicted,predicted_prob,'Doc2Vec',learner_name,record_dict['cv'][learner_name])


    m_label_train=y_train
    m_label_test=y_val
    learner = LogisticRegression(multi_class='ovr',max_iter=1000)
    SubTrainProcess('LR',learner)
    learner = SGDClassifier()
    SubTrainProcess('SGD',learner)
    learner = RandomForestClassifier()
    SubTrainProcess('RF',learner)
    learner = KNeighborsClassifier()
    SubTrainProcess('KNN',learner)
    learner = SVC(probability=True)
    SubTrainProcess('SVM',learner)    


def text_prepare(text,not_related_words,rejected_words):
    text = text.lower() 
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@#+,;]') 
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    
    text = REPLACE_BY_SPACE_RE.sub(' ',text) 
    remove = str.maketrans('','',string.punctuation) 
    text = text.translate(remove)
    text = BAD_SYMBOLS_RE.sub('',text) 
    for not_related_word in not_related_words:
        if not_related_word in text.split():
            return []
    # text = get_words(text)   
    STOPWORDS = set(stopwords.words('english'))
    words = [w for w in replace_abbreviations(text).split() if w not in STOPWORDS and len(w)>2 and w not in rejected_words]
    clean_text = ' '.join(words)
    return clean_text

def DropRep(train_df):
    record_texts = []
    deleted_idx = []
    tweet_text_list = train_df['tweet_text']
    for text_idx,tweet_text in enumerate(tweet_text_list):
        if tweet_text not in record_texts:
            record_texts.append(tweet_text)
        else:
            deleted_idx.append(text_idx)
    print (str(len(deleted_idx)),' rows has deleted for repeating...')
    train_df = train_df.drop(deleted_idx,0)

    return train_df
        

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
    
    train_df = pd.read_csv('df.csv')
    train_df = DropRep(train_df)

    ### train_df['genres'] = [i.split() for i in train_df['genres']]
    
    train_df['tweet_text'] = TransformMerge([text_prepare(x,not_related_words,rejected_words) for x in train_df['tweet_text']],rejected_words)
    
    print (train_df.columns)
    
    validation = train_df.sample(frac=validation_portion)
    train = train_df.drop(list(validation.index))
    print (train.shape[0],train.shape[1])
    tags = train['label'].values
    tag_dic={}
    
    for tag in tags:        
        if tag not in tag_dic:
            tag_dic[tag]=1
        else:
            tag_dic[tag]+=1
    df_labels = pd.DataFrame(list(tag_dic.items()), columns=['tag', 'count']).sort_values(by = 'count',axis = 0,ascending = False)
    print('total num of labels:',len(df_labels))
    print (df_labels)

    def GetIndicatorDf(indicator,model,model_feature):
        # print (tfidf.get_feature_names())
        terms = model.get_feature_names()

        # sum tfidf frequency of each term through documents
        sums = model_feature.sum(axis=0)

        # connecting term to its sums frequency
        data = []
        for col, term in enumerate(terms):
            data.append( (term, sums[0,col] ))

        ranking = pd.DataFrame(data, columns=['term','rank'])
        ranking = ranking.sort_values('rank', ascending=False)
        print(ranking)
        ranking.to_csv('{}.csv'.format(indicator),index=False)

    X_train, y_train = train.tweet_text, train.label
    X_val, y_val = validation.tweet_text, validation.label
    cv = CountVectorizer(min_df=min_df,max_df=max_df,ngram_range=ngram_range,token_pattern= '(\S+)')
    feature = cv.fit_transform(X_train)
    print(feature.shape)
    GetIndicatorDf('cv',cv,feature)
   # max_features=10000
    tfidf = TfidfVectorizer(min_df=min_df,max_df=max_df,ngram_range=ngram_range,token_pattern= '(\S+)',norm='l2', )
    feature = tfidf.fit_transform(X_train)
    print(feature.shape)
    GetIndicatorDf('tfidf',tfidf,feature)
    
    # mlb = MultiLabelBinarizer(classes=sorted(tag_dic.keys()))
    # y_train = mlb.fit_transform(y_train)
    # # print (y_train)
    # y_val = mlb.fit_transform(y_val)
    # print(y_train.shape)
    # print(train.genres)

    
    
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

    # SVC_pipeline = Pipeline([
    #             ('tfidf', TfidfVectorizer(min_df=5,max_df=0.9,ngram_range=(1,2),token_pattern= '(\S+)')),
    #             ('clf', OneVsRestClassifier(SVC(kernel='linear',probability=True), n_jobs=1)),
    #         ])
 
    # SVC_pipeline.fit(X_train,y_train)
    # predicted = SVC_pipeline.predict(X_val)
    # predicted_prob = SVC_pipeline.predict_proba(X_val)
    # print (predicted_prob[:10])
    # print_evaluation_scores(y_val,predicted,predicted_prob)

    def TrainProcess(nlp_mode,learner_name,learner):
        print (learner.get_params().keys())
        if learner_name == 'LR':
            parameters={
                # "clf__penalty":["l1","l2","elasticnet"],
                # "clf__scoring":["f1_macro"],
                # "clf__cv":[5],
                # "clf__solver":["saga","newton-cg","lbfgs"]
            }
        elif learner_name == 'SVM':
            parameters = {
                "clf__kernel": ["linear","poly","rbf"],
            }
        elif learner_name == 'KNN':
            parameters = {"clf__n_neighbors":[3,5,10,20,50]}
        elif learner_name == 'GB':
            parameters = {
                "clf__learning_rate": np.arange(0.01,0.15,0.05),
                "clf__n_estimators": range(100,1100,100),
                "clf__subsample":np.arange(0.1,0.6,0.1),
                "clf__min_samples_split":range(2,11)
            }
        elif learner_name == 'RF':
            parameters = {
                "clf__n_estimators": range(100,1100,100),
                "clf__criterion":["entropy","gini"]
            }
        else:
            parameters={}
        if nlp_mode == 'cv':
            algo_pipeline = Pipeline([
                        ('cv', cv),
                        ('clf', learner),
                    ])
        elif nlp_mode == 'tfidf':
            algo_pipeline = Pipeline([
                    ('tfidf', tfidf),
                    ('clf', learner),
                ])
        clf = RandomizedSearchCV(algo_pipeline,parameters,cv=5,scoring='accuracy')
        clf.fit(X_train,y_train)
        print (clf.best_params_)
        predicted = clf.predict(X_val)
        predicted_prob = clf.predict_proba(X_val)
        print (nlp_mode,learner)
        print (predicted_prob[0])
        print_evaluation_scores(y_val,predicted,predicted_prob,nlp_mode,learner_name,clf.best_params_)

    learner = LogisticRegression(multi_class='ovr',max_iter=1000)
    TrainProcess('cv','LR',learner)
    ### learner = GaussianProcessClassifier(max_iter_predict=1000)
    ### TrainProcess('cv','GP',learner)
    earner = SGDClassifier()
    TrainProcess('cv','SGD',learner)
    TrainProcess('tfidf','SGD',learner)
    learner = GradientBoostingClassifier()
    TrainProcess('cv','GB',learner)
    TrainProcess('tfidf','GB',learner)
    learner = RandomForestClassifier()
    TrainProcess('cv','RF',learner)
    TrainProcess('tfidf','RF',learner)
    learner = MultinomialNB()
    TrainProcess('cv','NB',learner)
    TrainProcess('tfidf','NB',learner)
    learner = KNeighborsClassifier()
    TrainProcess('cv','KNN',learner)
    TrainProcess('tfidf','KNN',learner)
    learner = SVC(probability=True)
    TrainProcess('cv','SVM',learner)
    TrainProcess('tfidf','SVM',learner)

    Doc2VecModel()

    out_json_dict = json.dumps(record_dict,indent=4)
    f3 = open(out_json_path, 'w')
    f3.write(out_json_dict)
    f3.close()
    
    
   




