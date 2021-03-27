### 1.sample & encode; 2.encode & standardized; 
### one-class SVM; K Means;  __ dimension reduction (LDA; PCA; ARD; lightGBM, FANS; )
### data analysis: All; 
### dimension reduction: Luna; Bingji; 


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from dateutil.parser import parse
import datetime
from datetime import timedelta
import math
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()  #transform features by scalling each feature to a given range
ohe_period = preprocessing.OneHotEncoder(handle_unknown='ignore') #Encode categorical features as a one-hot numeric array.  ignore if an unknown categorical feature is present during transform 

import category_encoders as ce
import lightgbm as lgb
import pickle  
from sklearn.metrics import roc_auc_score  
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
smo = SMOTE(random_state=42)
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


# PyTorch sampler ImbalancedDatasetSample: 
# - rebalance the class distributions when sampling from the imbalanced dataset 
# - estimate the sampling weights automatically 
# - avoid creating a new balanced dataset 
# - mitigate overfitting when it is used in conjunction with data augmentation techniques.
## Reference: https://github.com/ufoym/imbalanced-dataset-sampler
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """
    Samples elements randomly from a given list of indices for imbalanced dataset
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
    
    # feature2 = [datetime.datetime.strptime(str(i).split('.')[0],'%Y%m')  if not math.isnan(i)
    #     else i  for i in df[feature2_str]]
    # if method == 'past':
    #     Interval = [int((feature1[i]-feature2[i]).days/30) 
    #         if type(feature2[i]) != float  
    #         else feature2[i] for i in range(df.shape[0])]
    # elif method == 'future':
    #     Interval = [int((feature2[i]-feature1[i]).days/30) 
    #         if type(feature2[i]) != float  
    #         else feature2[i] for i in range(df.shape[0])]

    feature2 = [
        [
            int(str(i).split('.')[0][:4]), 
            int(str(i).split('.')[0][4:6])
        ] if not math.isnan(i) 
        else i 
        for i in df[feature2_str]
    ]
    if method == 'past':
        Interval = [
            (feature1[i].year-feature2[i][0])*12 + (feature1[i].month-feature2[i][1]) 
            if type(feature2[i]) != float  #🚩why would feature2[i] be a float?
            else feature2[i] 
            for i in range(df.shape[0])
        ]
    elif method == 'future':
        Interval = [
            (feature2[i][0]-feature1[i].year)*12 + (feature2[i][1]-feature1[i].month)
            if type(feature2[i]) != float  
            else feature2[i] 
            for i in range(df.shape[0])
        ]
    return Interval
    
def Turn2Binary(df,dimension_name,category1):
    return [1 if i ==category1 else 0 for i in (df[dimension_name]) ]

def Turn2Datetime(df,dimension_name):
    return [parse(str(i)) if not math.isnan(i) else i for i in df[dimension_name]] #turn a string to a date/time stamp

def PortionHighCategoricalVariable(train_data,test_data,labels): #not used in the script
    categories_columns = []
    for col in train_data.columns:
        if  '_id' in col:
            categories_columns.append(col)
    labels_2,train_data_2 = PortionSampling(labels,train_data,2)
    train_data_2[categories_columns] = train_data_2[categories_columns].fillna('nan')
    train_data[categories_columns] = train_data[categories_columns].fillna('nan')
    test_data[categories_columns] = test_data[categories_columns].fillna('nan')
    encoder = ce.TargetEncoder(cols=categories_columns,handle_unknown='ignore',  handle_missing='ignore')
    encoder.fit(train_data_2, labels_2)
    encoded_train = encoder.transform(train_data)
    try:
        encoded_test = encoder.transform(test_data)
    except:
        ### see which olc is missing in test dataset
        print ('*'*10)
        print (train_data.columns.difference(test_data.columns))
    for categories_column in categories_columns:
        temp_array = np.array(encoded_train[categories_column])
        encoded_train[categories_column] = min_max_scaler.fit_transform(temp_array.reshape(-1, 1))
    del_cols = []
    for categories_column in categories_columns:
        temp_array = np.array(encoded_test[categories_column])
        percent = len([i for i in temp_array if math.isnan(i)])/len(temp_array)
        if len([i for i in temp_array if math.isnan(i)])/len(temp_array) > 0.2:
            del_cols.append(categories_column)
        else:
            temp_median = np.nanmedian(temp_array)
            temp_array = [temp_median if math.isnan(i) else i for i in temp_array]
            encoded_test[categories_column] = min_max_scaler.fit_transform(np.array(temp_array).reshape(-1, 1))
    return encoded_train,encoded_test,del_cols

def HighCategoricalVariable(train_data,test_data,labels):
    ### claim_cause_fire too few
    categories_columns = ['claim_postal_code','claim_vehicle_id','claim_vehicle_brand','policy_holder_id',
        'policy_holder_expert_id','driver_id','driver_postal_code','driver_expert_id','driver_vehicle_id','third_party_1_id',
        'third_party_1_vehicle_id','third_party_1_expert_id','third_party_2_id','third_party_2_vehicle_id','third_party_2_expert_id',
        'repair_id','repair_postal_code','policy_holder_postal_code','policy_coverage_type','third_party_2_injured',
        'third_party_2_vehicle_type','third_party_2_form','third_party_2_country','driver_country','repair_country','repair_form',
        'claim_language','third_party_2_postal_code','third_party_2_year_birth','repair_year_birth','claim_date_occured_year',
        'claim_hour_occured','policy_coverage_1000','third_party_1_postal_code']
    train_data[categories_columns] = train_data[categories_columns].fillna('nan')
    test_data[categories_columns] = test_data[categories_columns].fillna('nan')
    # !!! more methods could apply
    encoder = ce.TargetEncoder(cols=categories_columns,handle_unknown='ignore',  handle_missing='ignore')
    encoder.fit(train_data, labels)
    encoded_train = encoder.transform(train_data)
    try:
        encoded_test = encoder.transform(test_data)
    except:
        ### see which olc is missing in test dataset
        print ('*'*10)
        print (train_data.columns.difference(test_data.columns))
    
    return encoded_train,encoded_test

def NumericaVariable(df_transform):
    numerica_columns = ['claim_vehicle_date_inuse','claim_vehicle_cyl','claim_vehicle_load','claim_vehicle_power',
        'policy_holder_year_birth','driver_year_birth','third_party_1_year_birth','policy_date_start',
        'policy_date_next_expiry','policy_date_last_renewed','policy_num_changes','policy_num_claims',
        'policy_premium_100','policy_coverage_1000','claim_date_interval']
    for detect_col in list(df_transform.columns):
        if np.any(np.isnan(df_transform[detect_col])) and detect_col not in numerica_columns:
            numerica_columns.append(detect_col)
    # min_max_scaler = preprocessing.MinMaxScaler()

    ### !!! a problem here is that do we need to standardize the HighCategoricalVariable
    for numerica_column in numerica_columns:
        temp_array = np.array(df_transform[numerica_column])
        temp_median = np.nanmedian(temp_array)
        df_transform[numerica_column] = [temp_median if math.isnan(i) else i for i in temp_array]
        df_transform[numerica_column] = min_max_scaler.fit_transform(np.array(df_transform[numerica_column]).reshape(-1, 1))
    return df_transform

def OneHotPreprocess(df_transform):
    ### claim_cause_fire too few
    ONEHOT_COLUMNS = ['claim_cause','policy_holder_form','driver_form','policy_holder_country',
        'claim_alcohol','claim_vehicle_type','claim_vehicle_fuel_type','third_party_1_injured',
        'third_party_1_vehicle_type','third_party_1_form','third_party_1_country','claim_date_occured_month']
    for ONEHOT_COLUMN in ONEHOT_COLUMNS:
        for idx in list(df_transform.index)[:100]:
            if type(df_transform[ONEHOT_COLUMN][idx]) == np.float64:
                df_transform[ONEHOT_COLUMN] = [str(i).split('.')[0] for i in df_transform[ONEHOT_COLUMN]]
                break
    local_ohe = LocalOneHotEncoder(ONEHOT_COLUMNS)
    df_transform[ONEHOT_COLUMN] = df_transform[ONEHOT_COLUMN].fillna('nan')
    local_ohe.fit(df_transform)
    oht_df = local_ohe.transform(df_transform)
    oht_df.index = list(df_transform.index)
    df_transform = pd.concat((df_transform,oht_df),axis=1)
    df_transform.drop(ONEHOT_COLUMNS,axis=1,inplace=True)

    return df_transform

## The following function is mainly used to transform the time stamp into time interval and return the processed dataframe.
def Preprocess(df_transform,the_year): 
    ### turn to date
    claim_date_registered = Turn2Datetime(df_transform,'claim_date_registered')
    claim_date_occured = Turn2Datetime(df_transform,'claim_date_occured')
    ### turn to claim_date_interval
    claim_date_interval = [(claim_date_registered[i]-claim_date_occured[i]).days for i in range(df_transform.shape[0])]
    df_transform['claim_date_interval'] =  claim_date_interval
    ### turn to interval by past date
    for time_interval in ['claim_vehicle_date_inuse','policy_date_start']:
        df_transform[time_interval] = GetIntervalMonths(claim_date_occured,time_interval,df_transform,'past')
    #the above for loop replace original claim_vehicle_date_inuse by the time interval between the claim occured date and inuse date
    #same for the policy_data_start
    
    ### turn to interval by future date
    # 'policy_date_next_expiry' and 'policy_date_last_renewed' are time for the future.
    for time_interval in ['policy_date_next_expiry','policy_date_last_renewed']:
        df_transform[time_interval] = GetIntervalMonths(claim_date_occured,time_interval,df_transform,'future')
    
    ### to get year and month for categorization
    df_transform ['claim_date_occured_year'] =  [i.year for i in claim_date_occured]
    df_transform ['claim_date_occured_month'] =  [i.month for i in claim_date_occured]

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
            claim_time_occured[i] == 0  # 🚩 why not append to claim_data_occured_hour?
        elif len(sub_item) == 4:
            claim_date_occured_hour.append(int(sub_item[0:2]))
    df_transform['claim_hour_occured'] = claim_date_occured_hour
    ### drop the processed columns
    df_transform.drop(["claim_date_registered", "claim_date_occured",'claim_time_occured'], axis=1,inplace=True)
    # - "claim_date_registered" has been transformed into "claim_time interval" 
    # - "claim_date_occured" has been transformed to year and month
    # - 'claim_time_occured' has been transformed to "claim_hour_occured"

    ### get their age
    for col in ['policy_holder_year_birth','driver_year_birth','third_party_1_year_birth','third_party_2_year_birth','repair_year_birth']:
        df_transform[col] = [the_year-i if not math.isnan(i) else i for i in df_transform[col]]
    
    ### Turn2Binary
    # ,'driver_injured','repair_sla'
    for col in ['claim_police','claim_liable']:
        df_transform[col] = Turn2Binary(df_transform,col,'Y')
    ### drop unimportant features
    # df_transform.drop([i for i in df_transform.columns if 'third_party_3' in i],axis=1,inplace=True)

    return df_transform

def OutlierDetection(encoded_train_with_val,label_train_with_val,encoded_test,label_test,pattern):
    
    # pca = decomposition.PCA()
    # pca.fit(encoded_train_with_val)  
    # total_var = sum(pca.explained_variance_)
    # var_list = [round(i/total_var,2) for i in pca.explained_variance_]
    # # print (var_list)
    # print ([(i,round(sum(var_list[:i])/sum(var_list),2)) for i in range(1,len(var_list)+1)])
    # n_components = 8
    # encoded_train_with_val_new = pca.transform(encoded_train_with_val.values)[:,:n_components]
    # encoded_test_new = pca.transform(encoded_test.values)[:,:n_components]
    
    # pos_ratio
     ## method 1
    # label_train_with_val_2,encoded_train_with_val_2 = PortionSampling(label_train_with_val,encoded_train_with_val,2)
    # print (encoded_train_with_val_2.columns.difference(encoded_test.columns))
    ### method 2
    # encoded_train_with_val_2, label_train_with_val_2 = smo.fit_resample(encoded_train_with_val_2, label_train_with_val_2)
    ### OneClassSVM
    # clf = OneClassSVM(nu=pos_ratio, gamma='scale').fit(encoded_train_with_val.values)
    ### IsolationForest
    ## fit(encoded_train_with_val,sample_weight=samples_weight)
    # samples_weight = SamplesWeight(label_train_with_val)
    
    if pattern == 'OneClassSVM':
        clf = OneClassSVM(nu=pos_ratio, gamma='scale').fit(encoded_train_with_val.values)
    elif pattern == 'IsolationForest':
        clf = IsolationForest(random_state=42,contamination=pos_ratio).fit(encoded_train_with_val.values)

    pred_train_prob = clf.decision_function(encoded_train_with_val.values)
    pred_test_prob =clf.decision_function(encoded_test.values)
    pred_train_prob = min_max_scaler.fit_transform(pred_train_prob.reshape(-1, 1))
    pred_train_prob = [1-i[0] for i in pred_train_prob]
    pred_test_prob = min_max_scaler.fit_transform(pred_test_prob.reshape(-1, 1))
    pred_test_prob = [1-i[0] for i in pred_test_prob]

    print ('OutlierDetection—{}:'.format(pattern))
    print (pred_test_prob[:10])    
    print ('train data: ',roc_auc_score(y_true=label_train_with_val,y_score=pred_train_prob))  
    print ('test data: ',roc_auc_score(y_true=label_test,y_score=pred_test_prob)) 
    threshold = 0.5
    labels_predict = [ 1 if pred > threshold else 0 for pred in pred_train_prob]
    print ('train dataset:')
    print(classification_report(label_train_with_val, labels_predict)) 
    labels_predict = [ 1 if pred > threshold else 0 for pred in pred_test_prob]
    print ('test dataset:')
    print(classification_report(label_test, labels_predict)) 
    return pred_test_prob


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
        voted_pro_predict = np.array(voted_pro_predict,dtype=float)
        final_pro_predict = np.sum(voted_pro_predict,axis=0)/voted_pro_predict.shape[0]
        return final_pro_predict,record_votes_list,record_lab_votes_list,item_dict
        
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


def FANS_Application(encoded_train_with_val,label_train_with_val,encoded_test,label_test):
    epoches = 2
    sample_method = 'bootstrap'
    iteration = 2
    output_list = []
    item_dict = {}

    ## method 1
    label_train_with_val_2,encoded_train_with_val_2 = PortionSampling(label_train_with_val,encoded_train_with_val,2)
    # print (encoded_train_with_val_2.columns.difference(encoded_test.columns))

    ### method 2
    encoded_train_with_val_2, label_train_with_val_2 = smo.fit_resample(encoded_train_with_val_2, label_train_with_val_2)
    ### original data
    # encoded_train_with_val_2, label_train_with_val_2 = encoded_train_with_val, label_train_with_val

    for item in list(encoded_train_with_val_2.columns):
        item_dict[item] = []

    for epoch in range(epoches):
        FANS_ = FANS(encoded_train_with_val_2,label_train_with_val_2,encoded_test,iteration,sample_method)
        final_pro_predict,record_votes_list,record_lab_votes_list,item_dict = FANS_.Main(item_dict,sample_method)
        output_list.append(final_pro_predict)
    final_pred = []
    for i in zip(*output_list):
        final_pred.append(np.median(i))
    print ('FANS:')
    print (final_pred[:10])
    print ('test data: ',roc_auc_score(y_true=label_test,y_score=final_pred)) 
    threshold = 0.5
    labels_predict = [ 1 if pred > threshold else 0 for pred in final_pred]
    print ('test dataset:')
    print(classification_report(label_test, labels_predict)) 
    item_mean_dict = {}
    for item in list(encoded_train_with_val.columns):
        if item.split('_')[0] in ['word','char']:
            item_mean_dict[item.split('_')[-1]] = np.mean(item_dict[item])
        else:
            item_mean_dict[item] = np.mean(item_dict[item])
    item_mean_dict= dict(sorted(item_mean_dict.items(),key=lambda x:x[1],reverse=True))
    print (item_mean_dict)


def SamplesWeight(labels):
    class_sample_count = np.array(
    [len(np.where(labels == t)[0]) for t in np.unique(labels)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in labels])
    return samples_weight

def PortionSampling(label_train_with_val,encoded_train_with_val,ratio_):
    pos_idx = [i for i in range(len(label_train_with_val)) if label_train_with_val[i] == 1]
    encoded_train_with_val_pos = encoded_train_with_val.iloc[pos_idx,:]
    encoded_train_with_val_neg = encoded_train_with_val.iloc[[i for i in range(len(label_train_with_val)) if label_train_with_val[i] == 0],:]
    encoded_train_with_val_neg_sub = encoded_train_with_val_neg.sample(encoded_train_with_val_pos.shape[0]*ratio_)
    encoded_train_with_val_2 = encoded_train_with_val_pos.append(encoded_train_with_val_neg_sub).reset_index(drop = True)
    label_train_with_val_2 = [1] * encoded_train_with_val_pos.shape[0]
    label_train_with_val_2.extend([0] * encoded_train_with_val_neg_sub.shape[0])
    return label_train_with_val_2,encoded_train_with_val_2


def NeuralNetworks(encoded_train_with_val,label_train_with_val,encoded_test,label_test):

    p_numbers = encoded_train_with_val.shape[1]
    # num of neurons for hidden layer
    N_HIDDEN = 128
    
    net_label_test = torch.tensor(list(df_test['class']))

    net_test_data = torch.tensor(np.array(encoded_test), dtype=torch.float32)
    
    # encoded_train_with_val.to_excel(out_train_path,index=False)
    # encoded_test.to_excel(out_train_path.replace('train','test'),index=False)
    # exit()

    ### method 1 (different portions of pos dataset according to neg data)
    label_train_with_val_2,encoded_train_with_val_2 = PortionSampling(label_train_with_val,encoded_train_with_val,2)
    ### method 2: SMOTE (dont accept NaN) more method in slides could apply
    ### !!! Question: it works in such sotuation (high dimension; imbalanced samples)?
    
    encoded_train_with_val_2, label_train_with_val_2 = smo.fit_resample(encoded_train_with_val, label_train_with_val)
    # print (encoded_train_with_val_2)
    print (len(label_train_with_val_2))
    batch_size = max(1000,int(len(label_train_with_val_2)/10))
    
    net_train_data = torch.tensor(np.array(encoded_train_with_val), dtype=torch.float32)
    net_train_data_2 = torch.tensor(np.array(encoded_train_with_val_2), dtype=torch.float32)
    net_train_labels_2 = torch.LongTensor(np.array(label_train_with_val_2))

    ### change the parameters of NN here
    net = torch.nn.Sequential(
        torch.nn.Linear(p_numbers, N_HIDDEN),
        torch.nn.Dropout(0.2),   
        torch.nn.ReLU(),
        torch.nn.Linear(N_HIDDEN, N_HIDDEN),
        torch.nn.Dropout(0.2),   
        torch.nn.ReLU(),
        torch.nn.Linear(N_HIDDEN, 2),
    )

    ### WeightedRandomSampler (change the weights of samples by idx)
    torch_dataset = Data.TensorDataset(net_train_data_2,net_train_labels_2)
    samples_weight = SamplesWeight(label_train_with_val_2)
    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()
    sampler = Data.WeightedRandomSampler(samples_weight, len(samples_weight),replacement=True)
    # shuffle=True , sampler=sampler

    ### ImbalancedDatasetSampler, it is a individual method, if interested, check it above
    sampler = ImbalancedDatasetSampler(torch_dataset)
    ### minibatch
    train_loader = Data.DataLoader(dataset=torch_dataset,batch_size=batch_size,
        sampler=sampler)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01,weight_decay=0.01)
    ### !!! This loss_func is not okay here, BCEWithLogitsLoss might be solution
    loss_func = torch.nn.CrossEntropyLoss()
    # nn.BCEWithLogitsLoss(pos_weight=weights)

    epoches = 1000
    stop_epoches = 10
    last_roc_test = 0
    count_continue = 1
    count_interval = 1

    for epoch in range(epoches+1):
        for step, (x,y) in enumerate(train_loader):
            out = net(x) 
            loss = loss_func(out, y)

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

        
        prediction = torch.max(net(net_train_data_2), 1)[1]
        # print (prediction)
        pred_y = prediction
        target_y = net_train_labels_2
        train_accuracy = (pred_y == target_y).sum().item() / float(pred_y.size(0))

        net.eval()  
        pred_test = torch.max(net(net_test_data), 1)[1]
        target_y = net_label_test
        test_accuracy = (pred_test == target_y).sum().item() / len(pred_test)

        pred_train_prob = torch.nn.functional.softmax(net(net_train_data),dim=1)
        pred_train_prob = (pred_train_prob.detach().numpy())
        pred_train_prob = [i[1] for i in pred_train_prob]
        # print (pred_train_prob[:10])
        pred_test_prob = torch.nn.functional.softmax(net(net_test_data),dim=1)
        pred_test_prob = (pred_test_prob.detach().numpy())
        pred_test_prob = [i[1] for i in pred_test_prob]
        # print (pred_test_prob[:10])
        
        net.train()

        # to see if the training have bad effects on ROC
        roc_test = roc_auc_score(y_true=label_train_with_val,y_score=pred_train_prob)
        if roc_test - last_roc_test <= 0.01:
            count_continue = 1
        else:
            count_continue = 0
        if count_continue == 1:
            count_interval += 1
        else:
            count_interval = 0

        last_roc_test = roc_test

        if count_interval >= stop_epoches:
            print ('stop training, early stopping...')
            break
        
        if epoch % 100 == 0:
            print('Epoch: ', epoch, '| train data roc: %.4f' % roc_auc_score(y_true=label_train_with_val,y_score=pred_train_prob),
                    '| test data roc: %.2f' % roc_test)
    
    ## indicators
    print ('NN:')
    print (pred_test_prob[:10])    
    print ('train data: ',roc_auc_score(y_true=label_train_with_val,y_score=pred_train_prob))  
    print ('test data: ',roc_auc_score(y_true=label_test,y_score=pred_test_prob)) 
    threshold = 0.5
    labels_predict = [ 1 if pred > threshold else 0 for pred in pred_train_prob]
    print ('train dataset:')
    print(classification_report(label_train_with_val, labels_predict)) 
    labels_predict = [ 1 if pred > threshold else 0 for pred in pred_test_prob]
    print ('test dataset:')
    print(classification_report(label_test, labels_predict)) 


def LightGBMPreprocess(df_transform):
    toIDX_cols = ['claim_cause',  'claim_vehicle_brand', 'policy_holder_form',  'third_party_1_vehicle_type',
    'third_party_1_form', 'third_party_2_injured', 'third_party_2_vehicle_type', 
    'third_party_2_form', 'third_party_2_country', 'repair_form', 'repair_country', 'policy_coverage_type',
    # 'claim_postal_code', 'policy_holder_postal_code', 'driver_postal_code', 'third_party_1_postal_code', 'third_party_2_postal_code', 'repair_postal_code',
    'claim_vehicle_id', 'policy_holder_id', 'policy_holder_expert_id', 'driver_id', 'driver_expert_id', 'driver_vehicle_id', 'third_party_1_id', 
    'third_party_1_vehicle_id', 'third_party_1_expert_id', 'third_party_2_id', 'third_party_2_vehicle_id', 'third_party_2_expert_id', 'repair_id']

    for toIDX_col in toIDX_cols:
        values_dict = {}
        for idx,value in enumerate(list(set(df_transform[toIDX_col].values))):
            values_dict[value] = idx
        new_idx_list = []
        for sub_item in df_transform[toIDX_col]:    
            try:
                new_idx_list.append(values_dict[sub_item])
            except:
                new_idx_list.append(sub_item)
        df_transform[toIDX_col] = new_idx_list
    return df_transform

def PortionLabels(train_data_pos,sub_train_data_neg,feature):
    sub_label_train = []
    sub_label_train_idx = []
    sub_label_train_idx.extend(list(train_data_pos.index))
    sub_label_train_idx.extend(list(sub_train_data_neg.index))
    for i in sub_label_train_idx:
        if feature == 'claim_amount':
            sub_label_train.append(float(df_train[feature][i].replace(',','.')))
        else:
            sub_label_train.append(df_train[feature][i])
    # print (sub_label_train_idx)
    return sub_label_train

def LightGBM(train_data,test_data,val_data,df_train,label_train,label_test,label_val,pattern_feature,pattern_sample,output_pattern,W_train,W_val):
    ### LightGBMPreprocess: retain NaN and simple make categories to rank numbers
    if pattern_feature == 'unfilled':
        train_data = LightGBMPreprocess(train_data)
        test_data = LightGBMPreprocess(test_data)
        val_data = LightGBMPreprocess(val_data)

    total_preds = []
    total_train_preds = []
    total_val_preds = []
    importance_dict = {}
    # for various portions of negative samples accroding to positive samples
    # Attention! it may preceed the number of negative samples (esp. it is not Sampling Without Replacement)
    portion_list =[1]*10
    portion_list.extend(list(np.arange(1.1,2.1,0.1)))
    if pattern_sample == 'portion':
        train_data_pos = train_data[df_train['class'] == 1]
        train_data_neg = train_data[df_train['class'] == 0]
    for idx_,portion in enumerate(portion_list):
        print (idx_+1)
        if pattern_sample == 'portion':
        ### Get the new datasets with different portions of negative datasets
            sub_train_data_neg = train_data_neg.sample(int(len(train_data_pos)*portion))
            train_data_neg = train_data_neg.drop(list(sub_train_data_neg.index))
            sub_label_train = PortionLabels(train_data_pos,sub_train_data_neg,'class')
            sub_weight_train = PortionLabels(train_data_pos,sub_train_data_neg,'claim_amount')
            sub_train_data = train_data_pos.append(sub_train_data_neg)
            test_data.to_csv('./test_1.csv')
            sub_train_data.to_csv('./train_1.csv')
            print (len(sub_label_train),train_data_pos.shape[0],sub_train_data_neg.shape[0])

        # X, val_X, y, val_y = train_test_split(  
        # sub_train_data,  
        # sub_label_train,  
        # test_size=0.1,  
        # random_state=1,  
        # stratify=sub_label_train 
        # )  
            X_train = sub_train_data  
            y_train = sub_label_train 
            W_train =  sub_weight_train
        else:
            X_train = train_data
            y_train = label_train 
        X_test = val_data  
        y_test = label_val  

        # create dataset for lightgbm  
        print (W_train)
        print (W_val)
        lgb_train = lgb.Dataset(X_train, y_train,weight=W_train)  
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train,weight=W_val)  
        # specify your configurations as a dict  
        params = {  
            'boosting_type': 'gbdt',  
            'objective': 'binary',  
            'metric':  'auc',  
            'num_leaves': 16,  ### could change but be careful about overfitting
            'max_depth': 8,  ### could change but be careful about overfitting
            # 'min_data_in_leaf': 450,  
            'learning_rate': 0.01,  ### could change but be careful about local optimization
            'feature_fraction': 0.5,  ### like random forest for its features to sample
            'bagging_fraction': 0.5,  ### like random forest for its samples to sample
            'bagging_freq': 100,  ### how many times for sample
            'lambda_l1': 0.01,    ### L1 norm (lead to more zero coeff)
            'lambda_l2': 0.01,    ### L2 norm
            # 'weight_column':'name:claim_amount',
            'is_unbalance': True # Note: while enabling this should increase the overall performance metric of your model, it will also result in poor estimates of the individual class probabilities
            }  

        # train  
        # print ('Start training...'+str(i))  
        gbm = lgb.train(params,  
                        lgb_train,  
                        num_boost_round=1500,   # max training epoches
                        valid_sets=lgb_eval, 
                        early_stopping_rounds=1000) # to which epoch to check early_stopping)  

        # print('Start predicting...')  

        preds_train = gbm.predict(train_data, num_iteration=gbm.best_iteration) 
        total_train_preds.append(preds_train)
        preds_val = gbm.predict(val_data, num_iteration=gbm.best_iteration) 
        total_val_preds.append(preds_val)
        importance = gbm.feature_importance()  
        preds = gbm.predict(test_data, num_iteration=gbm.best_iteration) 
        total_preds.append(preds)
        importance = gbm.feature_importance()  
         # lgb.plot_importance(gbm, max_num_features=30)
        # plt.title("Featurertances")
        # plt.show()
        
        names = gbm.feature_name()  
        # to collect the importance of features
        for index, im in enumerate(importance):  
            if names[index] not in importance_dict.keys():
                importance_dict[names[index]] = [im]
            else:
                importance_dict[names[index]].append(im)
    # for pred in preds:  
    #     result = 1 if pred > threshold else 0
    final_train_pred = []
    final_pred = []
    final_val_pred = []
    ### get the final median propobility for different portions of pos data sampling
    for i in zip(*total_preds):
        final_pred.append(round(np.median(i),4))
    for i in zip(*total_train_preds):
        final_train_pred.append(round(np.median(i),4))
    for i in zip(*total_val_preds):
        final_val_pred.append(round(np.median(i),4))
    # min_max_scaler = preprocessing.MinMaxScaler()
    # final_train_pred = list(min_max_scaler.fit_transform(np.array(final_train_pred).reshape(-1, 1)))
    # final_pred = list(min_max_scaler.fit_transform(np.array(final_pred).reshape(-1, 1)))
    ### the indicators
    print ('lightGBM:')
    print (final_pred[:10])
    print ('train data: ',roc_auc_score(y_true=label_train,y_score=final_train_pred))  
    threshold = 0.5  
    labels_predict = [ 1 if pred > threshold else 0 for pred in final_train_pred]
    print ('train dataset:')
    print(classification_report(label_train, labels_predict)) 
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
        df_output.to_csv(output_test_path,index=False)
        print ('val data: ',roc_auc_score(y_true=label_val,y_score=final_val_pred)) 
        exit()
    print ('test data: ',roc_auc_score(y_true=label_test,y_score=final_pred)) 
    
    labels_predict = [ 1 if pred > threshold else 0 for pred in final_pred]
    print ('test dataset:')
    print(classification_report(label_test, labels_predict)) 

    return final_pred

def GetSplitDataframe(df,portion):
    df_positive = df[df['class'] == 1]
    df_negative = df[df['class'] == 0]
    df_test = df_positive.sample(frac=portion).append(df_negative.sample(frac=portion)) # random selection of 30%of postive +30% of negative
    df_train = df.drop(list(df_test.index)) #data excluding the test data

    return df_train,df_test

def ClearID(train_data,val_data,test_data):
    del_cols = []
    for col in train_data.columns:
        if  '_id' not in col: 
            continue #skip the columns that does not have "_id"
        #take all the column names which does contain _id
        temp_list = list((set([i for i in list(train_data[col]) if type(i) == str])))  #i is the value in each column, it is making sure the value is string (skipping the empty cells)
        temp_list_2 = list([i for i in list(test_data[col]) if type(i) == str]) #🚩 why not adding set() in this line?
        if len(temp_list_2) == 0: #if there are all empty cells in the column, then the column will be marked to remove.
            del_cols.append(col)
        elif len([i for i in temp_list_2 if i in temp_list])/len(temp_list_2) > 0.2: 
        #the number of id name in test set matches with the id name in training set / the number of id name in test set >0.2
        # if there are more than 20% of datapoints in the test set that can be matched with training data, then drop the column, 🚩why?
            del_cols.append(col)
    
    for sub_df in [train_data,val_data,test_data]:
        sub_df = sub_df.drop(del_cols, 1) #🚩 I don't get why to drop those cols?
        #sub_de.drop(del_cols,1,inplace=True) could also work 

    return  train_data,val_data,test_data  #return the dataset that has deleted certain "_id" columns
    
def NewPreProcess(df):
    unimportant_features = ['claim_alcohol','claim_vehicle_type','claim_vehicle_fuel_type','policy_holder_country',
        'driver_form','driver_country','driver_injured','third_party_1_injured','third_party_1_country','repair_sla',
        'policy_num_claims']
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


if __name__ == '__main__':
    
    # 0. Load data and data preprocessing
    ### read data
    input_train_path = './train.csv'
    input_test_path = './test.csv'
    output_test_path = './test_output.csv'
    out_train_path = './encoded_train_with_val.xlsx'
    out_train_path_1 = './encoded_train_with_val.csv'
    out_train_path_2 = './encoded_test.csv'
    df = pd.read_csv(input_train_path,sep = ';',index_col='claim_id')
    # 45622 is deleted due to data quality of claim_vehicle_load (500) 🚩how did you detect it ?
    df = df.drop([45622], 0)
    df['class'] = [1 if i == 'Y' else 0 for i in df['fraud']]
    df = NewPreProcess(df) # 🚩 why does this function mean?
    
    # if "claim_amount" in df.columns:
    #     df["claim_amount"] = [float(str.replace(amount, ",", ".")) for amount in df["claim_amount"]]
    
    df_positive = df[df['class'] == 1]
    pos_ratio = df_positive.shape[0] / df.shape[0]
    test_data_portion = 0.3
    val_data_portion = 0.05
    ### to expand the positive samples by resample (2 times positive samples)
    expand_train_pos_num = 2
    df = df.reset_index()
    ### Get Split Dataframe
    df_train,df_test = GetSplitDataframe(df,test_data_portion) #create test (30% positive+ 30% negative) and first_training set (70% postive+70% negative)
    df_train,df_val = GetSplitDataframe(df_train,val_data_portion) # create validation set (0.7*0.05=3.5% postive +3.5% negative) and real training set (66.5% postive and 66.5% negative)
    # - Training data: 66.5% of positive +66.5% of negative;
    # - Test data: 30% of postive +30% of negative
    # - Validation data: 3.5% of postive +3.5% of negative
    df_train_pos = df_train[df_train['class'] == 1] # extract the postive case in training data
    # oversampling: to expand the positive samples
    for i in range(expand_train_pos_num):
        df_train = df_train.append(df_train_pos) # adding the postive case from the training set twice to the training set and generate an oversampling training set
    df_train = df_train.reset_index(drop = True)
    # get train , validation (prevent overfitting) and test datasets
    label_train = df_train['class'] 
    W_train = df_train['claim_amount']
    W_train = [float(i.replace(',','.')) for i in W_train] #transform xxx,xx to xxx.xx
    train_data = df_train.drop(["class", "fraud",'claim_amount'], 1) # or df_train.drop(columns=["class", "fraud",'claim_amount'])
    df_test = df_test.reset_index(drop = True) #drop=True means drop the original index
    label_test = list(df_test['class']) 
    test_data = df_test.drop(["class", "fraud",'claim_amount'], 1)
    label_val = df_val['class'] 
    W_val = df_val['claim_amount'] 
    W_val = [float(i.replace(',','.')) for i in W_val]
    val_data = df_val.drop(["class", "fraud",'claim_amount'], 1)

    ### standardized prepoess (drop unimportant features; cope with date kind features)
    # .drop('claim_id',axis=1)
    train_data = Preprocess(train_data,2017) #2017 is the claim registration year
    test_data = Preprocess(test_data,2017)
    val_data = Preprocess(val_data,2017)

    train_data,val_data,test_data = ClearID(train_data,val_data,test_data)


    # 1.LightGBM (no need for categorization and cope with NaN value)
    final_pred_LightGBM = LightGBM(train_data,test_data,val_data,df_train,label_train,label_test,label_val,'unfilled','portion','train',W_train,W_val)
    
    ### predict true test dataset
    df_test = pd.read_csv(input_test_path,sep = ',',index_col='claim_id')
    df_test = NewPreProcess(df_test)
    df = pd.read_csv(input_train_path,sep = ';',index_col='claim_id')
    ## 45622 is deleted due to data quality of claim_vehicle_load (500)
    df = df.drop([45622], 0)
    df['class'] = [1 if i == 'Y' else 0 for i in df['fraud']]
    df = NewPreProcess(df)
    df_train,df_val = GetSplitDataframe(df,0.1)
    train_data = Preprocess(df_train,2017)
    val_data = Preprocess(df_val,2017)
    label_train = df_train['class']
    W_train = df_train['claim_amount']
    W_train = [float(i.replace(',','.')) for i in W_train]
    train_data = df_train.drop(["class", "fraud",'claim_amount'], 1)
    label_val = df_val['class']
    W_val = df_val['claim_amount']
    W_val = [float(i.replace(',','.')) for i in W_val]
    val_data = df_val.drop(["class", "fraud",'claim_amount'], 1)
    test_data = Preprocess(df_test,2018)
    label_test = []

    train_data,val_data,test_data = ClearID(train_data,val_data,test_data)

    LightGBM(train_data,test_data,val_data,df_train,label_train,label_test,label_val,'unfilled','portion','predict',W_train,W_val)
    exit()
    ### for NN and FANS we have auto validation set split (so train dataset combined with validation dataset) 
    ### and they dont accept NaN value and need serious categorization
    train_data_with_val = train_data.append(val_data)
    train_data_with_val = train_data_with_val.reset_index(drop=True)
    label_train_with_val = label_train.append(label_val)
    label_train_with_val = list(label_train_with_val)
    ### First OneHotPreprocess for simple categorical variable (it means less categories) then HighCategoricalVariable by category_encoders
    encoded_train,encoded_val = HighCategoricalVariable(OneHotPreprocess(train_data),OneHotPreprocess(val_data),label_train)
    encoded_train_with_val,encoded_test = HighCategoricalVariable(OneHotPreprocess(train_data_with_val),OneHotPreprocess(test_data),label_train_with_val)
    ### output to see the value
    # encoded_train_with_val.to_csv(out_train_path,index=False)
    ### fill the NaN for numerical dimension by median
    encoded_train_with_val = NumericaVariable(encoded_train_with_val)
    encoded_train = NumericaVariable(encoded_train)
    encoded_val = NumericaVariable(encoded_val)
    encoded_test = NumericaVariable(encoded_test)

    # encoded_train_with_val.to_csv(out_train_path_1,index=False)
    # encoded_test.to_csv(out_train_path_2,index=False)

    ### categorization and SMOTE for LightGBM. They lead to serious overfitting.
    # LightGBM(encoded_train,encoded_test,encoded_val,df_train,label_train,label_test,label_val,'filled','portion')
    # smo = SMOTE(random_state=42)
    # encoded_train_2, label_train_2 = smo.fit_resample(encoded_train, label_train)
    # LightGBM(encoded_train_2,encoded_test,encoded_val,df_train,label_train_2,label_test,label_val,'filled','SMOTE')

    # print (np.any(np.isnan(encoded_train_with_val)))
    # print (np.any(np.isnan(encoded_test)))
    # exit()

    ### 2.NN, need pytorch
    # NeuralNetworks(encoded_train_with_val,label_train_with_val,encoded_test,label_test)

    # 3.FANS
    # FANS_Application(encoded_train_with_val,label_train_with_val,encoded_test,label_test)

    # 4.One-Class SVM and IsolationForest
    # OutlierDetection(encoded_train_with_val,label_train_with_val,encoded_test,label_test,'OneClassSVM')
    final_pred_IsolationForest = OutlierDetection(encoded_train_with_val,label_train_with_val,encoded_test,label_test,'IsolationForest')

    
    
