from tqdm import tqdm
import json
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from dateutil.parser import parse
import datetime
from datetime import timedelta
import math
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
ohe_period = preprocessing.OneHotEncoder(handle_unknown='ignore')
### may have some problems (you could Google to figure it out)
import category_encoders as ce
import lightgbm as lgb
import pickle   
from sklearn.metrics import roc_auc_score,confusion_matrix,classification_report,precision_recall_fscore_support
from sklearn.model_selection import train_test_split,StratifiedKFold,RandomizedSearchCV
from imblearn.over_sampling import SMOTE
smo = SMOTE(random_state=0)
### go to website
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
from scipy import stats
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn import decomposition
from keras.models import Sequential,Model
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout
from keras.layers.embeddings import Embedding

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
    
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')   
        torch.save(model, 'finish_model.pkl')                
        self.val_loss_min = val_loss


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        # self.callback_get_label = callback_get_label

        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def callback_get_label(self,dataset, idx):
        #callback function used in imbalanced dataset loader.
        i, target = dataset[idx]
        return int(target)

    def _get_label(self, dataset, idx):
        if self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        else:
            raise NotImplementedError
           

    def __iter__(self):
        # return (self.indices[i] for i in torch.multinomial(
        #     self.weights, self.num_samples, replacement=True))
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples



class LocalOneHotEncoder(object):
    
  def __init__(self, target_columns):
    '''
    @param: target_columns --- To perform one-hot encoding column name list. 
    '''
    self.enc = ohe_period
    self.col_names = target_columns

  def fit(self, df):
    '''
    @param: df --- pandas DataFrame
    '''
    self.enc.fit_transform(df[self.col_names].fillna('nan').values)
    self.labels = np.array(self.enc.categories_).ravel()
    self.new_col_names = self.gen_col_names(df)

  def gen_col_names(self, df):
    '''
    @param:  df --- pandas DataFrame
    '''
    new_col_names = []
    for col in self.col_names:
      for val in df[col].unique():
        new_col_names.append("{}_{}".format(col, val))
    return new_col_names

  def transform(self, df):
     '''
     @param:  df --- pandas DataFrame
     '''
     return pd.DataFrame(data = self.enc.fit_transform(df[self.col_names].fillna('nan')).toarray(), 
                         columns = self.new_col_names, 
                         dtype=int) 

def GetIntervalMonths(feature1,feature2_str,df,method):

    feature2 = [[int(str(i).split('.')[0][:4]),int(str(i).split('.')[0][4:6])] if not math.isnan(i)
        else i  for i in df[feature2_str]]
    if method == 'past':
        Interval = [(feature1[i].year-feature2[i][0])*12 + (feature1[i].month-feature2[i][1])
        if type(feature2[i]) != float  
        else feature2[i] for i in range(df.shape[0])]
    elif method == 'future':
        Interval = [(feature2[i][0]-feature1[i].year)*12 + (feature2[i][1]-feature1[i].month)
        if type(feature2[i]) != float  
        else feature2[i] for i in range(df.shape[0])]

    return Interval
    
def Turn2Binary(df,dimension_name,category1):
    return [1 if i ==category1 else 0 for i in (df[dimension_name]) ]
def Turn2Datetime(df,dimension_name):
    return [parse(str(i)) if not math.isnan(i) else i for i in df[dimension_name]]

def NumericaVariable(df_transform,del_cols):
    numerica_columns = ['claim_vehicle_date_inuse','claim_vehicle_cyl','claim_vehicle_load','claim_vehicle_power',
        'policy_holder_year_birth','driver_year_birth','third_party_1_year_birth','policy_date_start',
        'policy_date_next_expiry','policy_date_last_renewed','policy_num_changes','policy_num_claims',
        'policy_premium_100','policy_coverage_1000','claim_date_interval','third_party_2_year_birth', 'repair_year_birth']
    numerica_columns = [i for i in numerica_columns if i not in unimportant_features and i not in del_cols]
    for detect_col in list(df_transform.columns):
        if np.any(np.isnan(df_transform[detect_col])) and detect_col not in numerica_columns:
            numerica_columns.append(detect_col)
    # min_max_scaler = preprocessing.MinMaxScaler()

    ### !!! a problem here is that do we need to standardize the HighCategoricalVariable
    # for column_ in numerica_columns:
    # for column_ in list(df_transform.columns):
    for column_ in numerica_columns:
        temp_array = np.array(df_transform[column_])
        temp_median = np.nanmedian(temp_array)
        df_transform[column_] = [temp_median if math.isnan(i) else i for i in temp_array]
    for column_ in list(df_transform.columns):    
        df_transform[column_] = min_max_scaler.fit_transform(np.array(df_transform[column_]).reshape(-1, 1))
    return df_transform,df_transform[numerica_columns]

def HighCategoricalVariable(train_data,test_data,labels,del_cols):
    ### claim_cause_fire too few
    categories_columns = ['claim_postal_code','claim_vehicle_id','claim_vehicle_brand','policy_holder_id',
    'policy_holder_expert_id','driver_id','driver_postal_code','driver_expert_id','driver_vehicle_id','third_party_1_id',
    'third_party_1_vehicle_id','third_party_1_expert_id','third_party_2_id','third_party_2_vehicle_id','third_party_2_expert_id',
    'repair_id','repair_postal_code','policy_holder_postal_code','policy_coverage_type','third_party_2_injured',
    'third_party_2_vehicle_type','third_party_2_form','third_party_2_country','driver_country','repair_country','repair_form',
    'claim_language','third_party_2_postal_code','claim_date_occured_year',
    'claim_hour_occured','third_party_1_postal_code','third_party_1_vehicle_type','claim_cause','_third_party_3_withinfo']
    categories_columns = [i for i in categories_columns if i not in unimportant_features and i not in del_cols]
    train_data[categories_columns] = train_data[categories_columns].fillna('nan')
    test_data[categories_columns] = test_data[categories_columns].fillna('nan')
    # !!! more methods could apply
    # ce.TargetEncoder  CatBoostEncoder
    encoder = ce.CatBoostEncoder(cols=categories_columns,handle_unknown='ignore',  handle_missing='ignore')
    encoder.fit(train_data, labels)
    encoded_train = encoder.transform(train_data)
    try:
        encoded_test = encoder.transform(test_data)
    except:
        ### see which olc is missing in test dataset
        print ('*'*10)
        print (train_data.columns.difference(test_data.columns))
        print (test_data.columns.difference(train_data.columns))
    
    return encoded_train,encoded_test

def OneHotPreprocess(df_transform,del_cols):
    ### claim_cause_fire too few
    ONEHOT_COLUMNS = ['policy_holder_form','driver_form','policy_holder_country',
        'claim_alcohol','claim_vehicle_type','claim_vehicle_fuel_type','third_party_1_injured',
        'third_party_1_form','third_party_1_country','claim_date_occured_month']
    ONEHOT_COLUMNS = [i for i in ONEHOT_COLUMNS if i not in unimportant_features and i not in del_cols]
    for ONEHOT_COLUMN in ONEHOT_COLUMNS:
        for idx in list(df_transform.index)[:100]:
            if type(df_transform[ONEHOT_COLUMN][idx]) == float or type(df_transform[ONEHOT_COLUMN][idx]) == int:
                df_transform[ONEHOT_COLUMN] = [str(i).split('.')[0] for i in df_transform[ONEHOT_COLUMN]]
                break
            else:
                df_transform[ONEHOT_COLUMN] = [str(i)for i in df_transform[ONEHOT_COLUMN]]
    local_ohe = LocalOneHotEncoder(ONEHOT_COLUMNS)
    df_transform[ONEHOT_COLUMN] = df_transform[ONEHOT_COLUMN].fillna('nan')
    local_ohe.fit(df_transform)
    oht_df = local_ohe.transform(df_transform)
    oht_df.index = list(df_transform.index)
    df_transform = pd.concat((df_transform,oht_df),axis=1)
    df_transform.drop(ONEHOT_COLUMNS,axis=1,inplace=True)

    return df_transform

def NoNaNTransformation(sub_train_data,val_data,test_data,total_train_data,sub_label_train,del_cols):
    encoded_train,encoded_val = HighCategoricalVariable(OneHotPreprocess(sub_train_data,del_cols),OneHotPreprocess(val_data,del_cols),sub_label_train,del_cols)
    encoded_train,encoded_test = HighCategoricalVariable(OneHotPreprocess(sub_train_data,del_cols),OneHotPreprocess(test_data,del_cols),sub_label_train,del_cols)
    encoded_train,encoded_total_train = HighCategoricalVariable(OneHotPreprocess(sub_train_data,del_cols),OneHotPreprocess(total_train_data,del_cols),sub_label_train,del_cols)

    encoded_train = NumericaVariable(encoded_train,del_cols)[0]
    encoded_val = NumericaVariable(encoded_val,del_cols)[0]
    encoded_test = NumericaVariable(encoded_test,del_cols)[0]
    encoded_total_train = NumericaVariable(encoded_total_train,del_cols)[0]
    deletes = []
    
    for detect_col in list(encoded_train.columns):
        if detect_col not in list(encoded_test.columns):
            encoded_test[detect_col] = [0] * encoded_test.shape[0]
        if detect_col not in list(encoded_total_train.columns):
            encoded_total_train[detect_col] = [0] * encoded_total_train.shape[0]
        if detect_col not in list(encoded_val.columns):
            encoded_val[detect_col] = [0] * encoded_val.shape[0]
        if np.any(np.isnan(encoded_train[detect_col])) or np.any(np.isnan(encoded_test[detect_col])):
            deletes.append(detect_col)
    for detect_col in list(encoded_test.columns):
        if detect_col not in list(encoded_train.columns):
            encoded_train[detect_col] = [0] * encoded_train.shape[0]
        if detect_col not in list(encoded_total_train.columns):
            encoded_total_train[detect_col] = [0] * encoded_total_train.shape[0]
        if detect_col not in list(encoded_val.columns):
            encoded_val[detect_col] = [0] * encoded_val.shape[0]
    encoded_train = encoded_train.drop(deletes,1)
    encoded_total_train = encoded_total_train.drop(deletes,1)
    encoded_val = encoded_val.drop(deletes,1)
    encoded_test = encoded_test.drop(deletes,1)
    print (encoded_train.shape[1],encoded_val.shape[1],encoded_test.shape[1],encoded_total_train.shape[1])
    return encoded_train,encoded_val,encoded_test,encoded_total_train


def Preprocess(df_transform,the_year):
    ### turn to date
    claim_date_registered = Turn2Datetime(df_transform,'claim_date_registered')
    claim_date_occured = Turn2Datetime(df_transform,'claim_date_occured')
    ### turn to claim_date_interval
    claim_date_interval = [(claim_date_registered[i]-claim_date_occured[i]).days for i in range(df_transform.shape[0])]
    df_transform.loc[:,'claim_date_interval'] =  claim_date_interval
    ### turn to interval by past date
    for time_interval in ['claim_vehicle_date_inuse','policy_date_start']:
        df_transform.loc[:,time_interval]= GetIntervalMonths(claim_date_occured,time_interval,df_transform,'past')
    ### turn to interval by future date
    for time_interval in ['policy_date_next_expiry','policy_date_last_renewed']:
        df_transform.loc[:,time_interval] = GetIntervalMonths(claim_date_occured,time_interval,df_transform,'future')
    ### to get year and month for categorization
    df_transform.loc[:,'claim_date_occured_year'] =  [i.year for i in claim_date_occured]
    df_transform.loc[:,'claim_date_occured_month'] =  [int(i.month) for i in claim_date_occured]

    ### to get the hour info
    claim_time_occured = list(df_transform ['claim_time_occured'])
    claim_date_occured_hour =  []
    for i in range(df_transform.shape[0]):
        if math.isnan(claim_time_occured[i]):
            claim_date_occured_hour.append(claim_time_occured[i])
            continue
        sub_item = str(claim_time_occured[i]).split('.')[0]
        if len(sub_item) in [1,2]:
            claim_date_occured_hour.append(0)
        elif len(sub_item) == 3:
            claim_date_occured_hour.append(int(sub_item[0:1]))
        elif sub_item[0:1] == '24':
            claim_time_occured[i] == 0
        elif len(sub_item) == 4:
            claim_date_occured_hour.append(int(sub_item[0:2]))
    df_transform.loc[:,'claim_hour_occured'] = claim_date_occured_hour
    ### drop the processed columns
    df_transform.drop(["claim_date_registered", "claim_date_occured",'claim_time_occured'], axis=1,inplace=True)
    ### get their age
    for col in ['policy_holder_year_birth','driver_year_birth','third_party_1_year_birth','third_party_2_year_birth','repair_year_birth']:
        df_transform.loc[:,col] = [the_year-i if not math.isnan(i) else i for i in df_transform[col]]
    ### Turn2Binary
    # ,'driver_injured','repair_sla'
    for col in ['claim_police','claim_liable']:
        df_transform.loc[:,col] = Turn2Binary(df_transform,col,'Y')
    ### drop unimportant features
    # df_transform.drop([i for i in df_transform.columns if 'third_party_3' in i],axis=1,inplace=True)
    return df_transform



class FANS(object):
    
    def __init__(self,train_data,train_labels,test_data,iteration,sample_method,
                 bw_method='silverman',CS=np.arange(0.1,10,0.1),threshold=0.005):
        self.train_data = train_data
        self.train_labels = np.array(train_labels)
        self.test_data = np.array(test_data)
        self.train_num = self.train_data.shape[0]
        self.test_num = self.test_data.shape[0]
        self.feature_num = self.train_data.shape[1]
        assert self.train_data.shape[1] == self.test_data.shape[1]
        self.bw_method = bw_method
        self.iteration = iteration
        self.CS = CS
        self.threshold = threshold
        self.sample_method = sample_method
        
    def Main(self,item_dict,sample_method):
        voted_pro_predict = []
        voted_pro_train = []
        record_store_threhold = []
        record_votes_list = []
        record_lab_votes_list = []

        if sample_method == 'bootstrap':
            sample_df_list =[]
            # sample_idx_list = []
            for t in range(self.iteration):
                sample_df = self.train_data.sample(frac=1.0, replace=True)
                sample_df_list.append(sample_df)
                # sample_idx_list.extend(list(set(sample_df.index)))
                # print (len(list(set(sample_df.index))))

        for t in range(self.iteration):
#             print ('iteration ',t+1)
            if sample_method == 'bootstrap':
                train_data_sample = sample_df_list[t]
            else:
                train_data_sample = self.train_data.sample(frac=1/self.iteration)
                self.train_data = self.train_data.drop(list(train_data_sample.index),0)
            train_labels_sample = []
            for i in list(train_data_sample.index):
                train_labels_sample.append(self.train_labels[i])
            train_labels_sample = np.array(train_labels_sample)
            train_data_sample =np.array(train_data_sample)
            KFOLD = model_selection.KFold(n_splits=2,shuffle=True)
            for KDE_part,PLR_part in KFOLD.split(train_data_sample):
                X_KDE,Y_KDE = train_data_sample[KDE_part],train_labels_sample[KDE_part]
                X_PLR,Y_PLR = train_data_sample[PLR_part],train_labels_sample[PLR_part]
                transformed_X_PLR = self.CON_TRANER(X_KDE,Y_KDE,X_PLR)
                transformed_X_test = self.CON_TRANER(X_KDE,Y_KDE,self.test_data)
                PLR_CLF = LogisticRegressionCV(penalty='l1',Cs=self.CS,cv=3,solver='liblinear',max_iter=1e4)
                PLR_CLF.fit(transformed_X_PLR,Y_PLR)
                beta_list = PLR_CLF.coef_[0]
                for i_idx,item in enumerate(list(self.train_data.columns)):
                    item_dict[item].append(beta_list[i_idx])
                PLR_pro_train = PLR_CLF.predict_proba(transformed_X_PLR)[:,1]
                PLR_pro_predict = PLR_CLF.predict_proba(transformed_X_test)[:,1]
                PLR_label_predict = PLR_CLF.predict(transformed_X_test)
                record_lab_votes_list.append(PLR_label_predict)
                voted_pro_predict.append(PLR_pro_predict)
                voted_pro_train.append(PLR_pro_train)
        voted_pro_predict = np.array(voted_pro_predict,dtype=float)
        voted_pro_train = np.array(voted_pro_train,dtype=float)
        final_pro_train = np.sum(voted_pro_train,axis=0)/voted_pro_train.shape[0]
        final_pro_predict = np.sum(voted_pro_predict,axis=0)/voted_pro_predict.shape[0]
        return final_pro_predict,final_pro_train,record_votes_list,record_lab_votes_list,item_dict
        
    def CON_TRANER(self,X_KDE,Y_KDE,X_PLR):
        transformed_data = []
        X_PLR = X_PLR.T
        KDE_Generator = self.CON_KDE_Generator(X_KDE,Y_KDE)
        modify = lambda pro:np.max([self.threshold,pro])
        for item in X_PLR:
            item_set = set(item)
            pro_dict = {}
            KDE_0,KDE_1 = next(KDE_Generator)
            for i in item_set:
                pro_0,pro_1 = KDE_0(i),KDE_1(i)
                pro_0 = np.array(list(map(modify,pro_0)))
                pro_1 = np.array(list(map(modify,pro_1)))
                pro_dict[i] = np.log(pro_1/pro_0)
            map_pro = lambda value:pro_dict[value]
            item = np.array(list(map(map_pro,item)))[:,0]
            transformed_data.append(item)
        return np.array(transformed_data).T
    
    def CON_KDE_Generator(self,Xs,Ys):
        for item in range(self.feature_num):
            s_0 = Xs[Ys==0][:,item]
            s_1 = Xs[Ys==1][:,item]
            KDE_0 = self.KDE(s_0)
            KDE_1 = self.KDE(s_1)
            yield KDE_0,KDE_1
    
    def KDE(self,X):
        try:
            return stats.gaussian_kde(X,bw_method=self.bw_method)
        except:
            X += np.random.randn(X.shape[0])*0.1
            return stats.gaussian_kde(X,bw_method=self.bw_method)

def FANS_Application(model_name,sub_train_data,total_train_data,test_data,label_test,val_data,sub_label_train,total_label_train,label_val,output_pattern,sub_W_train,W_val,record_dict,del_cols):
    
    encoded_train,encoded_val,encoded_test,encoded_total_train = NoNaNTransformation(sub_train_data,val_data,test_data,total_train_data,sub_label_train,del_cols)
    epoches = 3
    sample_method = 'bootstrap'
    iteration = 3
    output_list = []
    train_list = []
    item_dict = {}
    encoded_train = encoded_train.reset_index(drop = True)
    # encoded_train.to_csv('./debug_1.csv')
    # encoded_test.to_csv('./debug_2.csv')

    for item in list(encoded_train.columns):
        item_dict[item] = []

    for epoch in range(epoches):
        FANS_ = FANS(encoded_train,sub_label_train,encoded_test,iteration,sample_method)
        final_pro_predict,final_pro_train,record_votes_list,record_lab_votes_list,item_dict = FANS_.Main(item_dict,sample_method)
        output_list.append(final_pro_predict)
        train_list.append(final_pro_train)
    final_train_pred = []
    final_pred = []
    for i in zip(*output_list):
        final_pred.append(np.median(i))
    for i in zip(*train_list):
        final_train_pred.append(np.median(i))
    record_dict[model_name]['total_train_preds'].append(final_train_pred)
    record_dict[model_name]['total_val_preds'].append([label_val])
    record_dict[model_name]['total_test_preds'].append(final_pred)
    item_mean_dict = {}
    for item in list(encoded_train.columns):
        if item.split('_')[0] in ['word','char']:
            item_mean_dict[item.split('_')[-1]] = np.mean(item_dict[item])
        else:
            item_mean_dict[item] = np.mean(item_dict[item])
    item_mean_dict= dict(sorted(item_mean_dict.items(),key=lambda x:x[1],reverse=True))

    for item in list(encoded_train.columns):
        record_dict[model_name]['importance_dict'][item] = item_dict[item]

    return record_dict

def OutlierDetection(model_name,sub_train_data,total_train_data,test_data,label_test,val_data,sub_label_train,total_label_train,label_val,output_pattern,sub_W_train,W_val,record_dict,del_cols):
    
    # pca = decomposition.PCA()
    # pca.fit(encoded_train_with_val)  
    # total_var = sum(pca.explained_variance_)
    # var_list = [round(i/total_var,2) for i in pca.explained_variance_]
    # # print (var_list)
    # print ([(i,round(sum(var_list[:i])/sum(var_list),2)) for i in range(1,len(var_list)+1)])
    # n_components = 8
    # encoded_train_with_val_new = pca.transform(encoded_train_with_val.values)[:,:n_components]
    # encoded_test_new = pca.transform(encoded_test.values)[:,:n_components]

    encoded_train,encoded_val,encoded_test,encoded_total_train = NoNaNTransformation(sub_train_data,val_data,test_data,total_train_data,sub_label_train,del_cols)
    
     ## method 1
    # label_train_with_val_2,encoded_train_with_val_2 = PortionSampling(label_train_with_val,encoded_train_with_val,2)
    # print (encoded_train_with_val_2.columns.difference(encoded_test.columns))
    ### method 2
    # encoded_train_with_val_2, label_train_with_val_2 = smo.fit_resample(encoded_train_with_val_2, label_train_with_val_2)
    ### OneClassSVM
    # clf = OneClassSVM(nu=pos_ratio, gamma='scale').fit(encoded_train_with_val.values)
    ### IsolationForest
    ## fit(encoded_train_with_val,sample_weight=samples_weight)
    pos_ratio = sum(total_label_train) / len(total_label_train)
    if model_name == 'OneClassSVM':
        clf = OneClassSVM(nu=pos_ratio, gamma='scale').fit(encoded_total_train.values)
    elif model_name == 'IsolationForest':
        clf = IsolationForest(random_state=42,contamination=pos_ratio).fit(encoded_total_train.values)

    pred_train_prob = clf.decision_function(encoded_total_train.values)
    pred_train_prob = min_max_scaler.fit_transform(np.array([0.5-i for i in pred_train_prob]).reshape(-1, 1))
    pred_train_prob = [i[0] for i in pred_train_prob]
    pred_test_prob =clf.decision_function(encoded_test.values)
    pred_test_prob = min_max_scaler.fit_transform(np.array([0.5-i for i in pred_test_prob]).reshape(-1, 1))
    pred_test_prob = [i[0] for i in pred_test_prob]

    print ('OutlierDetection—{}:'.format(model_name))
    print (pred_test_prob[:10])    
    print ('train data: ',roc_auc_score(y_true=total_label_train,y_score=pred_train_prob))  
    record_dict[model_name]['total_train_preds'].append(pred_train_prob)
    record_dict[model_name]['total_test_preds'].append(pred_test_prob)
    if output_pattern== 'train':
        print ('test data: ',roc_auc_score(y_true=label_test,y_score=pred_test_prob)) 
        record_dict[model_name]['total_val_preds'].append(label_val)
    return record_dict

def NeuralNetworks(model_name,sub_train_data,total_train_data,test_data,label_test,val_data,sub_label_train,total_label_train,label_val,output_pattern,sub_W_train,W_val,record_dict,del_cols):
    
    encoded_train,encoded_val,encoded_test,encoded_total_train = NoNaNTransformation(sub_train_data,val_data,test_data,total_train_data,sub_label_train,del_cols)
    p_numbers = encoded_train.shape[1]
    # num of neurons for hidden layer
    N_HIDDEN = 128

    net_label_val = torch.tensor(list(label_val))
    net_val_data = torch.tensor(np.array(encoded_val), dtype=torch.float32)
    
    net_label_test = torch.tensor(label_test)
    net_test_data = torch.tensor(np.array(encoded_test), dtype=torch.float32)
    
    # encoded_train_with_val.to_excel(out_train_path,index=False)
    # encoded_test.to_excel(out_train_path.replace('train','test'),index=False)
    # exit()

    # print (encoded_train_with_val_2)
    batch_size = len(sub_label_train)
    
    net_train_data = torch.tensor(np.array(encoded_total_train), dtype=torch.float32)
    net_train_data_2 = torch.tensor(np.array(encoded_train), dtype=torch.float32)
    net_train_labels_2 = torch.LongTensor(np.array(sub_label_train))

    ### change the parameters of NN here
    net = torch.nn.Sequential(
        torch.nn.Linear(p_numbers, N_HIDDEN),
        torch.nn.Dropout(0.3),   
        torch.nn.ReLU(),
        torch.nn.Linear(N_HIDDEN, N_HIDDEN),
        torch.nn.Dropout(0.3),   
        torch.nn.ReLU(),
        torch.nn.Linear(N_HIDDEN, 2),
    )

    ### WeightedRandomSampler (change the weights of samples by idx)
    torch_dataset = Data.TensorDataset(net_train_data_2,net_train_labels_2)
    # samples_weight = SamplesWeight(label_train_with_val_2)
    # samples_weight = torch.from_numpy(samples_weight)
    # samples_weigth = samples_weight.double()
    sampler = Data.WeightedRandomSampler(sub_W_train, len(sub_W_train),replacement=True)
    # shuffle=True , sampler=sampler

    ### ImbalancedDatasetSampler, it is a individual method, if interested, check it above
    # sampler = ImbalancedDatasetSampler(torch_dataset)
    ### minibatch
    train_loader = Data.DataLoader(dataset=torch_dataset,batch_size=batch_size,sampler=sampler)
    val_torch_dataset = Data.TensorDataset(net_val_data,net_label_val)
    val_loader = Data.DataLoader(dataset=val_torch_dataset)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01,weight_decay=0.01)
    ### !!! This loss_func is not okay here, BCEWithLogitsLoss might be solution
    # loss_func = torch.nn.CrossEntropyLoss()
    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight= torch.reshape(torch.tensor(sub_W_train),(-1,1)))
    loss_func = torch.nn.CrossEntropyLoss()

    epoches = 1000
    val_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.reshape(torch.tensor(W_val),(-1,1)))
    val_criterion = torch.nn.CrossEntropyLoss()
    patience = 20
    early_stopping = EarlyStopping(patience, verbose=False)

    for epoch in range(epoches):
        for step, (x,y) in enumerate(train_loader):
            out = torch.nn.functional.softmax(net(x),dim=1)
            # out.detach().numpy()
            # loss = loss_func(out, torch.Tensor([[0,1] if i == 1 else [1,0] for i in sub_label_train]))
            loss = loss_func(out, y)
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

        
        # prediction = torch.max(net(net_train_data_2), 1)[1]
        # # print (prediction)
        # pred_y = prediction
        # target_y = net_train_labels_2
        # train_accuracy = (pred_y == target_y).sum().item() / float(pred_y.size(0))

        # net.eval()  
        # pred_test = torch.max(net(net_test_data), 1)[1]
        # target_y = net_label_test
        # test_accuracy = (pred_test == target_y).sum().item() / len(pred_test)

        def NNGetProb(data):
            pred_prob = torch.nn.functional.softmax(net(data),dim=1)
            pred_prob = (pred_prob.detach().numpy())
            pred_prob = [i[1] for i in pred_prob]
            return pred_prob
        # print (pred_train_prob[:10])
        # pred_train_prob = NNGetProb(net_train_data)
        pred_train_prob = NNGetProb(net_train_data)
        pred_test_prob = NNGetProb(net_test_data)
        # print (pred_test_prob[:10])

        
        
        net.train()

        # to see if the training have bad effects on ROC
        
        # '| train data roc: %.4f' % roc_auc_score(y_true=total_label_train,y_score=pred_train_prob),
        if epoch % 100 == 0 and epoch != 0:
            try:
                print (pred_test_prob[:10])
                roc_test = roc_auc_score(y_true=label_test,y_score=pred_test_prob)
                print('Epoch: ', epoch, 
                        '| test data roc: %.2f' % roc_test)
            except:
                pass
        
        net.eval()
        pred_val_prob = NNGetProb(net_val_data)  
        # valid_loss = val_criterion(torch.nn.functional.softmax(net(net_val_data),dim=1), torch.Tensor([[0,1] if i == 1 else [1,0] for i in label_val]))	
        valid_loss = val_criterion(torch.nn.functional.softmax(net(net_val_data),dim=1), torch.LongTensor(list(label_val)))	

        early_stopping(valid_loss, net)
        if early_stopping.early_stop and epoch>epoches/10:
            print("Early stopping on epoch ",epoch+1)
            break
    
    ## indicators
    record_dict[model_name]['total_train_preds'].append(pred_train_prob)
    record_dict[model_name]['total_val_preds'].append(pred_val_prob)
    record_dict[model_name]['total_test_preds'].append(pred_test_prob)
    return record_dict


def LightGBMPreprocess(df_transform,val_data,test_data,total_train_data,del_cols):
    toIDX_cols = ['claim_cause',  'claim_vehicle_brand', 'policy_holder_form',  'third_party_1_vehicle_type',
    'third_party_1_form', 'third_party_2_injured', 'third_party_2_vehicle_type', 
    'third_party_2_form', 'third_party_2_country', 'repair_form', 'repair_country', 'policy_coverage_type',
    # 'claim_postal_code', 'policy_holder_postal_code', 'driver_postal_code', 'third_party_1_postal_code', 'third_party_2_postal_code', 'repair_postal_code',
    'claim_vehicle_id', 'policy_holder_id', 'policy_holder_expert_id', 'driver_id', 'driver_expert_id', 'driver_vehicle_id', 'third_party_1_id', 
    'third_party_1_vehicle_id', 'third_party_1_expert_id', 'third_party_2_id', 'third_party_2_vehicle_id', 'third_party_2_expert_id', 'repair_id',
    'claim_date_occured_month']

    toIDX_cols = [i for i in toIDX_cols if i not in unimportant_features and i not in del_cols]

    def GetNewIdxDf(df_fill,toIDX_col,values_dict):
        new_idx_list = []
        for sub_item in df_fill[toIDX_col]:    
            try:
                new_idx_list.append(values_dict[sub_item])
            except:
                if type(sub_item) == str:
                    new_idx_list.append(0)
                else:
                    new_idx_list.append(sub_item)
        df_fill[toIDX_col] = new_idx_list
        return df_fill

    for toIDX_col in toIDX_cols:
        values_dict = {}
        values_list = list(total_train_data[toIDX_col].values)
        values_list.extend(list(test_data[toIDX_col].values))
        values_list.extend(list(val_data[toIDX_col].values))
        values_list.extend(list(df_transform[toIDX_col].values))
        values_list = list(set(values_list))
        for idx,value in enumerate(values_list):
            if value not in values_dict.keys():
                values_dict[value] = idx+1
        
        df_transform = GetNewIdxDf(df_transform,toIDX_col,values_dict)
        val_data = GetNewIdxDf(val_data,toIDX_col,values_dict)
        test_data = GetNewIdxDf(test_data,toIDX_col,values_dict)
        total_train_data =  GetNewIdxDf(total_train_data,toIDX_col,values_dict)
                
    return df_transform,val_data,test_data,total_train_data

def LightGBM(model_name,sub_train_data,total_train_data,test_data,val_data,sub_label_train,total_label_train,label_val,output_pattern,sub_W_train,W_val,record_dict,del_cols):
    ### LightGBMPreprocess: retain NaN and simple make categories to rank numbers
   
    # for various portions of negative samples accroding to positive samples
    # Attention! it may preceed the number of negative samples (esp. it is not Sampling Without Replacement)
    # portion_list =[1]*10
    # portion_list.extend(list(np.arange(1.1,2.1,0.1)))
    
    sub_train_data,val_data,test_data,total_train_data = LightGBMPreprocess(sub_train_data,val_data,test_data,total_train_data,del_cols)
    X_train = sub_train_data  
    y_train = sub_label_train 
    W_train =  sub_W_train
    
    X_test = val_data  
    y_test = label_val  
    
    # create dataset for lightgbm  
    lgb_train = lgb.Dataset(X_train, y_train,weight=W_train)  
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train,weight=W_val)  
    # specify your configurations as a dict  
    params = {  
        'boosting_type': 'gbdt',  
        'objective': 'binary',  
        'metric':  'auc',  
        'verbose':-1,
        # 'num_leaves': 16,  ### could change but be careful about overfitting
        # 'max_depth': 8,  ### could change but be careful about overfitting
        # 'min_data_in_leaf': 450,  
        'learning_rate': 0.01,  ### could change but be careful about local optimization
        # 'feature_fraction': 0.6,  ### like random forest for its features to sample
        # 'bagging_fraction': 0.6,  ### like random forest for its samples to sample
        # 'bagging_freq': 200,  ### how many times for sample
        # 'lambda_l1': 0.01,    ### L1 norm (lead to more zero coeff)
        # 'lambda_l2': 0.01,    ### L2 norm
        # 'weight_column':'name:claim_amount',
        'is_unbalance': False # Note: while enabling this should increase the overall performance metric of your model, it will also result in poor estimates of the individual class probabilities
        }  

    # train  
    # print ('Start training...'+str(i))  
    
    rds_params = {
        'bagging_freq': range(100, 500, 100),
        'min_child_weight': range(3, 20, 2),
        'colsample_bytree': np.arange(0.4, 1.0),
        'max_depth': range(4, 32, 2),
        'num_leaves':range(16, 64, 4),
        # 'subsample': np.arange(0.5, 1.0, 0.1),
        'feature_fraction': np.arange(0.5, 1.0, 0.1),
        'bagging_fraction': np.arange(0.5, 1.0, 0.1),
        'lambda_l1': np.arange(0.01, 0.1, 0.01),
        'lambda_l2': np.arange(0.01, 0.1, 0.01),
        'min_child_samples': range(10, 30)}
    
    model = lgb.LGBMClassifier(**params)
    optimized_GBM = RandomizedSearchCV(model, rds_params, n_iter=50, cv=cv_search, n_jobs=4)
    optimized_GBM.fit(X_train, y_train) 
    print('best parameters:{0}'.format(optimized_GBM.best_params_))
    print('best score:{0}'.format(optimized_GBM.best_score_))
    params.update(optimized_GBM.best_params_)
    # print (params)
    gbm = lgb.train(params,  
                    lgb_train,  
                    num_boost_round=1500,   # max training epoches
                    valid_sets=lgb_eval, 
                    early_stopping_rounds=1000) # to which epoch to check early_stopping)  

    # print('Start predicting...')  
    
    preds_train = gbm.predict(total_train_data, num_iteration=gbm.best_iteration) 
    record_dict[model_name]['total_train_preds'].append(preds_train)

    preds_val = gbm.predict(val_data, num_iteration=gbm.best_iteration) 
    record_dict[model_name]['total_val_preds'].append(preds_val)
    importance = gbm.feature_importance()  
    preds = gbm.predict(test_data, num_iteration=gbm.best_iteration) 
    record_dict[model_name]['total_test_preds'].append(preds)
    importance = gbm.feature_importance()  
        # lgb.plot_importance(gbm, max_num_features=30)
    # plt.title("Featurertances")
    # plt.show()
    
    names = gbm.feature_name()  
    # to collect the importance of features
    for index, im in enumerate(importance):  
        if names[index] not in record_dict[model_name]['importance_dict'].keys():
            record_dict[model_name]['importance_dict'][names[index]] = [im]
        else:
            record_dict[model_name]['importance_dict'][names[index]].append(im)
    # for pred in preds:  
    #     result = 1 if pred > threshold else 0
    
    return record_dict


def PrintIndicators(final_pred,W_,label_):
    sort_idx_list = np.argsort(-np.array(final_pred))
    final_score = sum([W_[i] for i in sort_idx_list[:100] if label_[i] == 1])
    total_score = sum([W_[i] for i in sort_idx_list[:100]])
    total_ratio = round(final_score/total_score,4)
    print (final_score,total_ratio)
    return total_score,total_ratio

def ResultOutput(record_dict,test_data,model_name,label_train,label_test,W_train,W_test,output_pattern):
    print (model_name)
    total_test_preds = record_dict[model_name]['total_test_preds']
    total_train_preds = record_dict[model_name]['total_train_preds']
    importance_dict = record_dict[model_name]['importance_dict']
    final_train_pred = []
    final_pred = []
    final_val_pred = []
    ### get the final median propobility for different portions of pos data sampling
    for i in zip(*total_test_preds):
        final_pred.append(round(np.median(i),4))
    
    # for i in zip(*total_val_preds):
    #     final_val_pred.append(round(np.median(i),4))
    # min_max_scaler = preprocessing.MinMaxScaler()
    # final_train_pred = list(min_max_scaler.fit_transform(np.array(final_train_pred).reshape(-1, 1)))
    # final_pred = list(min_max_scaler.fit_transform(np.array(final_pred).reshape(-1, 1)))
    ### the indicators
    
    try:
        for i in zip(*total_train_preds):
            final_train_pred.append(round(np.median(i),4))
        train_auc = roc_auc_score(y_true=label_train,y_score=final_train_pred)
        print ('train data: ',train_auc)
        total_train_score,total_train_ratio = PrintIndicators(final_train_pred,W_train,label_train)  
    except:
        train_auc = 0
        total_train_score = 0
        total_train_ratio = 0

    
    # print ('val data: ',roc_auc_score(y_true=label_val,y_score=final_val_pred))  
    if model_name in ['lightGBM','FANS']:
        for key_ in importance_dict.keys():
            importance_dict[key_] = round(np.median(importance_dict[key_]),4)
        importance_dict = sorted(importance_dict.items(), key = lambda kv:(kv[1], kv[0]),reverse = True)
        print (importance_dict)
    

    if output_pattern == 'predict':
        print (pd.Series(final_pred).describe())
        test_data_idx = list(test_data.index)
        data = {'ID':test_data_idx,
        'PROB':final_pred}
        df_output = pd.DataFrame(data)
        # print(df_output)
        df_output.to_csv(output_test_path.format(model_name),index=False)
        return train_auc,total_train_score,total_train_ratio
    
    if output_pattern == 'train':
        threshold = 0.5
        labels_predict = [ 1 if pred > threshold else 0 for pred in final_pred]
        print ('test dataset:')
        print (final_pred[:10])
        print(classification_report(label_test, labels_predict)) 
        test_auc = roc_auc_score(y_true=label_test,y_score=final_pred)
        print ('test data: ',roc_auc_score(y_true=label_test,y_score=final_pred)) 
        total_test_score,total_test_ratio = PrintIndicators(final_pred,W_test,label_test)
        
        return test_auc,total_test_score,total_test_ratio
        

def ClearID(train_data,test_data):
    del_cols = []
    for col in train_data.columns:
        if  '_id' not in col:
            continue
        temp_list = list((set([i for i in list(train_data[col]) if type(i) == str])))
        temp_list_2 = list([i for i in list(test_data[col]) if type(i) == str])
        if len(temp_list_2) == 0:
            del_cols.append(col)
        elif len([i for i in temp_list_2 if i in temp_list])/len(temp_list_2) > 0.2:
            del_cols.append(col)
    train_data = train_data.drop(del_cols, 1) 
    test_data= test_data.drop(del_cols, 1) 
    return  train_data,test_data,del_cols

def GetPortionSplitDataframe(df,portion):
    df_positive = df[df['class'] == 1]
    df_negative = df[df['class'] == 0]
    df_test = df_positive.sample(frac=portion).append(df_negative.sample(frac=portion))
    df_train = df.drop(list(df_test.index))

    return df_train,df_test

def GetSplitDataframe(df,test_data_portion):
    ### Get Split Dataframe
    df = df.sample(frac=1.0)
    df_train,df_test = GetPortionSplitDataframe(df,test_data_portion)
    # df_train = df_train.reset_index()
    # df_train,df_val = GetPortionSplitDataframe(df_train,val_data_portion)
    
    # df_test = df_test.reset_index(drop = True)
    test_data,label_test,W_test = GetDataandLabel(df_test)

    label_train = df_train['class']
    # label_val = df_val['class']
    # W_val = df_val['claim_amount']
    # W_val = [float(i.replace(',','.')) for i in W_val]
    # val_data = df_val.drop(["class", "fraud",'claim_amount'], 1)

    return df_train,label_train,df_test,test_data,label_test,W_test
    
def NewPreProcess(df):
    
    df = df.drop(unimportant_features, 1)
    for col in df.columns:
        if col.startswith('third_party_3') and '_third_party_3_withinfo' not in df.columns:
            third_party_3_withinfo = []
            third_party_list = list(df[col])
            claim_num_third_parties = list(df['claim_num_third_parties'])
            for i,item in enumerate(third_party_list):
                # print (claim_num_third_parties[i],item)
                if claim_num_third_parties[i] >= 3: 
                    try:
                        if math.isnan(item):
                            third_party_3_withinfo.append(0)
                        else:
                            third_party_3_withinfo.append(1)
                    except:
                        third_party_3_withinfo.append(1)
                else:
                    third_party_3_withinfo.append(np.nan)
            df['_third_party_3_withinfo'] = third_party_3_withinfo
        if 'postal_code' in col:
            postal_code = list(df[col])
            postal_code = [int(str(i)[:2]) if not math.isnan(i) else i for i in postal_code]
            df[col] = postal_code
        if col.startswith('third_party_3'):
            df = df.drop([col], 1)
    return df

def GetDataandLabel(dataframe_):
    label_ = dataframe_['class']
    W_ = dataframe_['claim_amount']
    W_ = [float(i.replace(',','.')) for i in W_]
    dataframe_ = dataframe_.drop(["class", "fraud",'claim_amount'], 1)
    return dataframe_,label_,W_

def TrainProcess(df_train, label_train,test_data,label_test,W_test,model_names,output_pattern,final_record_dict,del_cols):

    record_dict = {}
    for model_name in model_names:
        record_dict[model_name] = {}
        record_dict[model_name]['total_test_preds'] = []
        record_dict[model_name]['total_train_preds'] = []
        record_dict[model_name]['total_val_preds'] = []
        record_dict[model_name]['importance_dict'] = {}
        
    df_train = df_train.reset_index(drop = True)
    df_train_2 = Preprocess(df_train.copy(),2017)
    total_label_train = list(df_train_2['class'])
    total_weight_train = [float(amount.replace( ",", ".")) for amount in list(df_train_2['claim_amount'])]
    total_train_data = df_train_2.drop(["class","fraud",'claim_amount'], 1)
    
    skf = StratifiedKFold(n_splits=k_num,shuffle=True)
    for train_index, val_index in skf.split(df_train, label_train):
        X_train, X_val = df_train.iloc[train_index], df_train.iloc[val_index]
        y_train, y_val = label_train.iloc[train_index], label_train.iloc[val_index]
        train_data = Preprocess(X_train,2017)
        X_val = Preprocess(X_val,2017)
        val_data,label_val,W_val = GetDataandLabel(X_val)

        train_data_pos = train_data[train_data['class'] == 1]
        ### to expand the positive samples
        
        for i in range(expand_train_pos_num):
            train_data = train_data.append(train_data_pos)

        train_data_pos = train_data[train_data['class'] == 1]
        train_data_neg = train_data[train_data['class'] == 0]

        sub_train_data_neg = train_data_neg.sample(int(len(train_data_pos)*portion_neg))
        sub_train_data = train_data_pos.append(sub_train_data_neg)
        # sub_label_train = PortionLabels(train_data_pos,sub_train_data_neg,'class')
        # sub_weight_train = PortionLabels(train_data_pos,sub_train_data_neg,'claim_amount')
        sub_train_data,sub_label_train,sub_W_train = GetDataandLabel(sub_train_data)
        print (len(sub_label_train),train_data_pos.shape[0],sub_train_data_neg.shape[0])

        
        model_name = 'lightGBM'
        record_dict = LightGBM(model_name,sub_train_data,total_train_data,test_data,val_data,sub_label_train,total_label_train,label_val,output_pattern,sub_W_train,W_val,record_dict,del_cols)
        
        model_name = 'NeuralNetworks'
        record_dict = NeuralNetworks(model_name,sub_train_data,total_train_data,test_data,list(label_test),val_data,sub_label_train,total_label_train,label_val,output_pattern,sub_W_train,W_val,record_dict,del_cols)

        model_name = 'FANS'
        record_dict = FANS_Application(model_name,sub_train_data,total_train_data,test_data,list(label_test),val_data,sub_label_train,total_label_train,label_val,output_pattern,sub_W_train,W_val,record_dict,del_cols)

        model_name = 'IsolationForest'
        record_dict = OutlierDetection(model_name,sub_train_data,total_train_data,test_data,list(label_test),val_data,sub_label_train,total_label_train,label_val,output_pattern,sub_W_train,W_val,record_dict,del_cols)

        model_name = 'OneClassSVM'
        record_dict = OutlierDetection(model_name,sub_train_data,total_train_data,test_data,list(label_test),val_data,sub_label_train,total_label_train,label_val,output_pattern,sub_W_train,W_val,record_dict,del_cols)

    print ('*'*10)
    print ('*'*10)
    print ('*'*10)
    print (output_pattern)
    for model_name in model_names:
        print ('*'*10)
        label_test = list(label_test)
        if model_name == 'FANS':
            test_auc,total_test_score,total_test_ratio = ResultOutput(record_dict,test_data,
            model_name,sub_label_train,label_test,sub_W_train,W_test,output_pattern)
        else:
            test_auc,total_test_score,total_test_ratio = ResultOutput(record_dict,test_data,
            model_name,total_label_train,label_test,total_weight_train,W_test,output_pattern)
        if output_pattern == 'train':
            final_record_dict[model_name]['test_auc'].append(test_auc)
            final_record_dict[model_name]['total_test_score'].append(total_test_score)
            final_record_dict[model_name]['total_test_ratio'].append(total_test_ratio)
        elif output_pattern == 'predict':
            final_record_dict[model_name]['train_auc'].append(test_auc)
            final_record_dict[model_name]['total_train_score'].append(total_test_score)
            final_record_dict[model_name]['total_train_ratio'].append(total_test_ratio)
    return final_record_dict

if __name__ == '__main__':
    
    ### read data
    ### etc
    input_train_path = './train.csv'
    input_test_path = './test.csv'
    output_test_path = './test_output_{}.csv'
    out_indicators_path = './out_indicators.json'
    out_indicators_train_path = './out_indicators_train.json'
    # out_train_path = './encoded_train_with_val.xlsx'
    # out_train_path_1 = './encoded_train_with_val.csv'
    # out_train_path_2 = './encoded_test.csv'
    model_names = ['lightGBM','NeuralNetworks','FANS','OneClassSVM','IsolationForest']
    # model_names = ['OneClassSVM','IsolationForest']
    ### to see how many epoches for trial on train dataset
    train_epoches = 15
    ### unimportant_features, directly deleted
    unimportant_features = ['claim_alcohol','claim_vehicle_type','claim_vehicle_fuel_type','policy_holder_country',
        'driver_form','driver_country','driver_injured','third_party_1_injured','third_party_1_country','repair_sla',
        'policy_num_claims']

    ### hyperparameters
    ## portion of test_data
    test_data_portion = 0.3
    ## hum many times to expand positive data
    expand_train_pos_num = 3
    ## hum many times for sampling negative data concerning positive data
    portion_neg = 1
    ## number of K folds
    k_num = 10
    ## number of K folds for RandomizedSearch
    cv_search = 2

    df = pd.read_csv(input_train_path,sep = ';',index_col='claim_id')
    # 45622 is deleted due to data quality of claim_vehicle_load (500)
    df = df.drop([45622], 0)
    df['class'] = [1 if i == 'Y' else 0 for i in df['fraud']]
    df = NewPreProcess(df)
    
    final_record_dict = {}
    for model_name in model_names:
        final_record_dict[model_name] = {}
        final_record_dict[model_name]['test_auc'] = []
        final_record_dict[model_name]['total_test_score'] = []
        final_record_dict[model_name]['total_test_ratio'] = []
    for epoch in tqdm(range(train_epoches)):
        ### get train , validation (prevent overfitting) and test datasets
        df_train,label_train,df_test,test_data,label_test,W_test = GetSplitDataframe(df,0.3)
        ### standardized prepoess (drop unimportant features; cope with date kind features)
        # .drop('claim_id',axis=1)
        df_train,test_data,del_cols = ClearID(df_train,test_data)
        final_record_dict = TrainProcess(df_train, label_train,Preprocess(test_data,2017),label_test,W_test,model_names,'train',final_record_dict,del_cols)
    out_indicators_dict = json.dumps(final_record_dict,indent=4)
    f2 = open(out_indicators_path, 'w')
    f2.write(out_indicators_dict)
    f2.close()
    for model_name in model_names:
        print ('*'*10)
        print (model_name)
        for key_ in final_record_dict[model_name].keys():
            print (key_,final_record_dict[model_name][key_])
            print (key_,np.median(final_record_dict[model_name][key_]))
    

    final_record_dict = {}
    for model_name in model_names:
        final_record_dict[model_name] = {}
        final_record_dict[model_name]['train_auc'] = []
        final_record_dict[model_name]['total_train_score'] = []
        final_record_dict[model_name]['total_train_ratio'] = []
    
    ### predict true test dataset
    df_test = pd.read_csv(input_test_path,sep = ',',index_col='claim_id')
    df_test = NewPreProcess(df_test)
    df = pd.read_csv(input_train_path,sep = ';',index_col='claim_id')
    ## 45622 is deleted due to data quality of claim_vehicle_load (500)
    df = df.drop([45622], 0)
    ### Get Split Dataframe
    # df_train,df_val = GetPortionSplitDataframe(df_train,val_data_portion)
    df['class'] = [1 if i == 'Y' else 0 for i in df['fraud']]
    df = NewPreProcess(df)
    skf = StratifiedKFold(n_splits=k_num,shuffle=True)
    label_train = df['class']

    df,df_test,del_cols = ClearID(df,df_test)
    final_record_dict = TrainProcess(df, label_train,Preprocess(df_test,2018),[1],[1],model_names,'predict',final_record_dict,del_cols)
    print (final_record_dict)
    out_indicators_train_dict = json.dumps(final_record_dict,indent=4)
    f3 = open(out_indicators_train_path, 'w')
    f3.write(out_indicators_train_dict)
    f3.close()
    exit()
    

    
    
