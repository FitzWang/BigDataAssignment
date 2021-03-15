# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 11:12:48 2021

@author: Fitz Wang
"""

import pandas as pd
import numpy as np
from category_encoders.target_encoder import TargetEncoder
from category_encoders.one_hot import OneHotEncoder
from category_encoders.count import CountEncoder

from sklearn import metrics

import matplotlib.pyplot as plt

class DataPreprocess():
    def __init__(self,data_train, data_test):
        self.data_train = data_train
        self.data_test = data_test
        self.num_fraud = np.sum(self.data_train["fraud"]=="Y")
        self.num_Notfraud = np.sum(self.data_train["fraud"]=="N")
        variable_train = set(self.data_train.columns)
        variable_test = set(self.data_test.columns)
        var_inTRnotTE = variable_train.difference(variable_test)
        var_inTEnotTR = variable_test.difference(variable_train)
        # General information
        print("General info:\n --------")
        print("Variable difference in training set and test set:\n In train not test: {0} \n In test not train:{1}".format(var_inTRnotTE,var_inTEnotTR))
        print("---------")
        print('Number of fraud is:{0} and number of non-fraud is:{1}'.format(self.num_fraud, self.num_Notfraud))
        assert self.num_fraud + self.num_Notfraud == len(self.data_train), "There're missing value on target (fraud or not)"
    
    # output weight used for sampling positive observations
    def ExtractAmount(self):
        data_amount = self.data_train[self.data_train['fraud'] == "Y"]
        self.data_amount = data_amount.loc[:,["claim_id","claim_amount"]]
        self.data_amount["claim_amount"] = self.data_amount["claim_amount"].apply(lambda x: float(x.split(',')[0]) + \
                                                                                  float(x.split(',')[1])/100)
        # # standardization and use sigmoid function to transform to probability
        # self.data_amount["claim_amount"] = (self.data_amount["claim_amount"] - self.data_amount["claim_amount"].mean())/\
        #     self.data_amount["claim_amount"].std()
        # self.data_amount["claim_amount"] = self.data_amount["claim_amount"].apply(lambda x: 1 / (1 + np.exp(-x)))
        
        # add log amount
        self.data_amount["log_amount"] = self.data_amount['claim_amount'].apply(np.log)
        # simply use amount as weight
        self.data_amount["claim_amount_prob"] = self.data_amount["claim_amount"]/self.data_amount["claim_amount"].sum()
    
    def ManualDrop(self):
        # Set claim_id as index
        self.data_train = self.data_train.set_index('claim_id')
        self.data_test = self.data_test.set_index('claim_id')
        
        ### Variables need to drop indentified manually
        # Variable not in the test set
        var_notinTest = ["claim_amount"]
        # Variable accosiated to third party
        var_thirdParty = ['third_party_1_id','third_party_1_postal_code',
                          'third_party_1_injured', 'third_party_1_vehicle_type',
                          'third_party_1_form', 'third_party_1_year_birth',
                          'third_party_1_country', 'third_party_1_vehicle_id',
                          'third_party_1_expert_id', 'third_party_2_id',
                          'third_party_2_postal_code', 'third_party_2_injured',
                          'third_party_2_vehicle_type', 'third_party_2_form',
                          'third_party_2_year_birth', 'third_party_2_country',
                          'third_party_2_vehicle_id', 'third_party_2_expert_id',
                          'third_party_3_id', 'third_party_3_postal_code',
                          'third_party_3_injured', 'third_party_3_vehicle_type',
                          'third_party_3_form', 'third_party_3_year_birth',
                          'third_party_3_country', 'third_party_3_vehicle_id',
                          'third_party_3_expert_id']
        # Variable accosiated to repair shop
        var_repair = ['repair_id','repair_form', 'repair_year_birth', 'repair_country']
        # Other variables can be dropped
        var_others = ['claim_time_occured',"driver_vehicle_id",'policy_date_last_renewed','repair_postal_code']

        self.data_train = self.data_train.drop(columns = var_notinTest + var_thirdParty + var_repair + var_others)
        self.data_test = self.data_test.drop(columns = var_thirdParty + var_repair + var_others)
    
    # transfer target value to 0 and 1
    def TransferTarget(self):
        self.data_train['fraud'] = self.data_train.fraud.map(dict(N=int(0), Y=int(1)))
        
        
    def DateProcess(self):
        def DateDiffDays(Date1,Date2):
            Date1 = Date1.apply(str)
            Date1 = pd.to_datetime(Date1,format='%Y%m%d')        
            Date2 = Date2.apply(str)
            Date2 = pd.to_datetime(Date2,format='%Y%m%d')        
            return abs((Date1 - Date2).dt.days)
        
        # Compute difference between claim resigtered day and occured day, and drop old variables
        self.data_train["claim_registered_occured_diff"] = DateDiffDays(self.data_train ["claim_date_registered"],
                                                                        self.data_train ["claim_date_occured"])
        self.data_test["claim_registered_occured_diff"] = DateDiffDays(self.data_test ["claim_date_registered"],
                                                                        self.data_test ["claim_date_occured"])
        self.data_train = self.data_train.drop(columns=["claim_date_registered","claim_date_occured"])
        self.data_test = self.data_test.drop(columns=["claim_date_registered","claim_date_occured"])
        
        # Compute ages (to 2017 or 2018) of claim vehicle (in years), and drop old variables
        self.data_train["claim_vehicle_age"] = abs(2017 - (self.data_train["claim_vehicle_date_inuse"][self.data_train["claim_vehicle_date_inuse"].notna()]/100).astype(int))
        self.data_test["claim_vehicle_age"] = abs(2018 - (self.data_test["claim_vehicle_date_inuse"][self.data_test["claim_vehicle_date_inuse"].notna()]/100).astype(int))
        self.data_train = self.data_train.drop(columns=["claim_vehicle_date_inuse"])
        self.data_test = self.data_test.drop(columns=["claim_vehicle_date_inuse"])
        
        # Compute ages (to 2017 or 2018) of policy holder (in years), and drop old variables
        self.data_train["police_holder_age"]  = abs(2017 - (self.data_train["policy_holder_year_birth"][self.data_train["policy_holder_year_birth"].notna()]).astype(int))
        self.data_test["police_holder_age"]  = abs(2018 - (self.data_test["policy_holder_year_birth"][self.data_test["policy_holder_year_birth"].notna()]).astype(int))
        self.data_train = self.data_train.drop(columns=["policy_holder_year_birth"])
        self.data_test = self.data_test.drop(columns=["policy_holder_year_birth"])
        
        # Compute ages (to 2017 or 2018) of driver (in years), and drop old variables
        self.data_train["driver_age"]  = abs(2017 - (self.data_train["driver_year_birth"][self.data_train["driver_year_birth"].notna()]).astype(int))
        self.data_test["driver_age"]  = abs(2018 - (self.data_test["driver_year_birth"][self.data_test["driver_year_birth"].notna()]).astype(int))
        self.data_train = self.data_train.drop(columns=["driver_year_birth"])
        self.data_test = self.data_test.drop(columns=["driver_year_birth"])
        
        
        # Compute month difference between policy starting date and next expiring date
        yearDiff = (self.data_train["policy_date_next_expiry"][self.data_train["policy_date_next_expiry"].notna()]/100).astype(int)\
            - (self.data_train["policy_date_start"][self.data_train["policy_date_start"].notna()]/100).astype(int)
        mod_monthDiff = (self.data_train["policy_date_next_expiry"][self.data_train["policy_date_next_expiry"].notna()]%100).astype(int)\
            - (self.data_train["policy_date_start"][self.data_train["policy_date_start"].notna()]%100).astype(int)
        self.data_train["policy_duration"] = yearDiff*12 + mod_monthDiff
        
        yearDiff = (self.data_test["policy_date_next_expiry"][self.data_test["policy_date_next_expiry"].notna()]/100).astype(int)\
            - (self.data_test["policy_date_start"][self.data_test["policy_date_start"].notna()]/100).astype(int)
        mod_monthDiff = (self.data_test["policy_date_next_expiry"][self.data_test["policy_date_next_expiry"].notna()]%100).astype(int)\
            - (self.data_test["policy_date_start"][self.data_test["policy_date_start"].notna()]%100).astype(int)
        self.data_test["policy_duration"] = yearDiff*12 + mod_monthDiff
        
        self.data_train = self.data_train.drop(columns=["policy_date_next_expiry","policy_date_start"])
        self.data_test = self.data_test.drop(columns=["policy_date_next_expiry","policy_date_start"])
    
    def IDProcess(self):
        # claim vehicle id counts as new variable
        self.data_train['freq_vehicle_id'] = self.data_train.groupby(["claim_vehicle_id"])["claim_cause"].transform('count')
        self.data_test['freq_vehicle_id'] = self.data_test.groupby(["claim_vehicle_id"])["claim_cause"].transform('count')
        self.data_train = self.data_train.drop(columns=["claim_vehicle_id"])
        self.data_test = self.data_test.drop(columns=["claim_vehicle_id"])
        
        # policy holder id counts as new variable
        self.data_train['freq_policy_holder_id'] = self.data_train.groupby(["policy_holder_id"])["claim_cause"].transform('count')
        self.data_test['freq_policy_holder_id'] = self.data_test.groupby(["policy_holder_id"])["claim_cause"].transform('count')
        
        # new variable indicates if policy holder id euquals to driver id
        self.data_train['policy_holder_is_driver'] = (self.data_train["policy_holder_id"] == self.data_train["driver_id"])
        self.data_test['policy_holder_is_driver'] = (self.data_test["policy_holder_id"] == self.data_test["driver_id"])
        
        self.data_train = self.data_train.drop(columns=["policy_holder_id"])
        self.data_test = self.data_test.drop(columns=["policy_holder_id"])
        self.data_train = self.data_train.drop(columns=["driver_id"])
        self.data_test = self.data_test.drop(columns=["driver_id"])
        
        # new variables to indicate if expert involved to analyze the damage, both policy holder and driver
        self.data_train['policy_holder_assigned_expert'] = self.data_train['policy_holder_expert_id'].notna()
        self.data_train['driver_assigned_expert'] = self.data_train['driver_expert_id'].notna()
        self.data_test['policy_holder_assigned_expert'] = self.data_test['policy_holder_expert_id'].notna()
        self.data_test['driver_assigned_expert'] = self.data_test['driver_expert_id'].notna()
        
        self.data_train = self.data_train.drop(columns=["policy_holder_expert_id"])
        self.data_train = self.data_train.drop(columns=["driver_expert_id"])
        self.data_test = self.data_test.drop(columns=["policy_holder_expert_id"])
        self.data_test = self.data_test.drop(columns=["driver_expert_id"])
    
    def PosCodeProcess(self):
        # use only first digit to represent claim post code (indicate province in Belgium)
        self.data_train["claim_postal_code"] = (self.data_train["claim_postal_code"][self.data_train["claim_postal_code"].notna()]/1000).astype(int)
        self.data_test["claim_postal_code"] = (self.data_test["claim_postal_code"][self.data_test["claim_postal_code"].notna()]/1000).astype(int)
        
        # use two digit to indicate policy holder post combined with country code
        self.data_train['policy_holder_country'] = self.data_train.policy_holder_country.map(dict(B=int(1), N=int(2)))
        self.data_test['policy_holder_country'] = self.data_test.policy_holder_country.map(dict(B=int(1), N=int(2)))
        
        self.data_train["policy_holder_postal_code"] = (self.data_train["policy_holder_postal_code"][self.data_train["policy_holder_postal_code"].notna()]/1000).astype(int)\
            + self.data_train['policy_holder_country']*10
        self.data_test["policy_holder_postal_code"] = (self.data_test["policy_holder_postal_code"][self.data_test["policy_holder_postal_code"].notna()]/1000).astype(int)\
            + self.data_test['policy_holder_country']*10
            
        self.data_train = self.data_train.drop(columns = ["policy_holder_country"])
        self.data_test = self.data_test.drop(columns = ["policy_holder_country"])
        
        # use two digit to indicate driver post combined with country code
        self.data_train['driver_country'] = self.data_train.driver_country.map(dict(B=int(1), N=int(2)))
        self.data_test['driver_country'] = self.data_test.driver_country.map(dict(B=int(1), N=int(2)))
        
        self.data_train["driver_postal_code"] = (self.data_train["driver_postal_code"][self.data_train["driver_postal_code"].notna()]/1000).astype(int)\
            + self.data_train['driver_country']*10
        self.data_test["driver_postal_code"] = (self.data_test["driver_postal_code"][self.data_test["driver_postal_code"].notna()]/1000).astype(int)\
            + self.data_test['driver_country']*10
            
        self.data_train = self.data_train.drop(columns = ["driver_country"])
        self.data_test = self.data_test.drop(columns = ["driver_country"])
            
    def OtherProcess(self):
        # fill nan in claim alcohol with negative, because in most cases, no test means it's not like drunk driving
        self.data_train["claim_alcohol"] = self.data_train["claim_alcohol"].fillna("N")
        self.data_test["claim_alcohol"] = self.data_test["claim_alcohol"].fillna("N")
        
        # fill nan in vehicle brand with others
        self.data_train["claim_vehicle_brand"] = self.data_train["claim_vehicle_brand"].fillna("others")
        self.data_test["claim_vehicle_brand"] = self.data_test["claim_vehicle_brand"].fillna("others")
    
    def BinaryEncode(self):
        YNList = ['claim_liable', 'claim_police', 'driver_injured', 'repair_sla']
        FMList = ['policy_holder_form','driver_form']
        for YNvar in YNList:
            self.data_train[YNvar] = self.data_train[YNvar].map(dict(Y=int(1), N=int(0)))
            self.data_test[YNvar] = self.data_test[YNvar].map(dict(Y=int(1), N=int(0)))
        
        for FMvar in FMList:
            self.data_train[FMvar] = self.data_train[FMvar].map(dict(M=int(1), F=int(0)))
            self.data_test[FMvar] = self.data_test[FMvar].map(dict(M=int(1), F=int(0)))
        
        self.data_train['claim_alcohol'] = self.data_train.claim_alcohol.map(dict(P=int(1), N=int(0)))
        self.data_test['claim_alcohol'] = self.data_test.claim_alcohol.map(dict(P=int(1), N=int(0)))
        
    # apply preprocess methods before split training set for cross validation
    def applyPreprocess(self, saveFile = False):
        self.ExtractAmount()
        self.TransferTarget()
        self.ManualDrop()
        self.DateProcess()
        self.IDProcess()
        self.PosCodeProcess()
        self.OtherProcess()
        self.BinaryEncode()       
        
        if saveFile == True:
            self.data_train.to_csv("train_preprocessed.csv")
            self.data_test.to_csv("train_preprocessed.csv")
            
    def TargetEncode(self, data_train, data_test, variables ,targetLabel = 'fraud'):
        # target-encode postal code
        TE_encoder = TargetEncoder(cols = variables)
        data_train[variables] = TE_encoder.fit_transform(data_train[variables], data_train[targetLabel])
        data_test[variables] = TE_encoder.transform(data_test[variables])        
        # encoder = ce.TargetEncoder(cols = variables, handle_unknown='ignore',  handle_missing='ignore')
        # data_train = encoder.fit_transform(data_train, data_train[targetLabel])
        # data_test = encoder.transform(data_test[variables])
        return data_train,data_test
    
    def OneHotEncode(self, data_train, data_test, variables):
        dataframe = pd.concat([data_train, data_test])
        OH_encoder = OneHotEncoder(cols = variables)
        transformed =  OH_encoder.fit_transform(dataframe)
        return transformed.iloc[:len(data_train)],transformed.iloc[len(data_train):].drop(columns = ['fraud'])

    def CountEncode(self, data_train, data_test, variables):
        CO_encoder = CountEncoder(cols = variables)
        return CO_encoder.fit_transform(data_train),CO_encoder.fit_transform(data_test)
    
    def applyEncoder(self):
        # encoding strategy right now
        targetEncoderList = ['claim_postal_code','policy_holder_postal_code','driver_postal_code','policy_coverage_type']
        self.data_train, self.data_test = self.TargetEncode(self.data_train, self.data_test, targetEncoderList)
        oneHotList = ['claim_cause','claim_vehicle_type','claim_language']
        self.data_train,self.data_test = data.OneHotEncode(self.data_train, self.data_test, oneHotList)
        countList = ['claim_vehicle_brand']
        self.data_train,self.data_test = data.CountEncode(self.data_train, self.data_test, countList)
    
    
    # sampling based on amount as probability
    def UnderSampling(self, dataFrame, timesPos, timesNeg2Pos, seed = 0):
        # make sure every pos case being sampled at once! e.g. samples = allpos + sampledpos
        num_pos = round(self.num_fraud * (timesPos))
        num_pos_sampled = round(self.num_fraud * (timesPos-1))
        np.random.seed(seed)
        samplePos_idx = np.random.choice(self.data_amount["claim_id"], num_pos_sampled, replace=True, p = self.data_amount["claim_amount_prob"])
        dataFrameSampled_pos = dataFrame.loc[samplePos_idx]
        dataFrame_pos = dataFrame[dataFrame['fraud']==1]
        
        dataFrame_neg = dataFrame[dataFrame['fraud']==0]
        num_neg = round(num_pos*timesNeg2Pos)
        np.random.seed(seed)
        sampleNeg_idx = np.random.choice(dataFrame_neg.index, num_neg, replace=False)
        dataFrameSampled_neg = dataFrame.loc[sampleNeg_idx]
        
        return pd.concat([dataFrameSampled_pos,dataFrame_pos,dataFrameSampled_neg])
    
    def OverSampling(self, dataFrame, timesPos2Neg, seed = 0):
        dataFrame_neg = dataFrame[dataFrame['fraud']==0]
        num_pos_sampled = round(timesPos2Neg*len(dataFrame_neg))
        np.random.seed(seed)
        samplePos_idx = np.random.choice(self.data_amount["claim_id"], num_pos_sampled, replace=True, p = self.data_amount["claim_amount_prob"])
        dataFrameSampled_pos = dataFrame.loc[samplePos_idx]
        
        return pd.concat([dataFrameSampled_pos,dataFrame_neg])
    
    def FillNan(self, dataFrame, method = 'median'):
        dfcopy = dataFrame.copy()
        columns_withNan = dataFrame.columns[dataFrame.isna().any()].tolist()
        # if method == 'median':
        #     for column in columns_withNan:
        dfcopy[columns_withNan] = dfcopy[columns_withNan].fillna(dfcopy[columns_withNan].median(skipna=True))
        return dfcopy

class MyModel():
    def __init__(self):
        pass
    
    def CalResult(self,test_label,predication,ROC = True):
        fpr, tpr, thresholds = metrics.roc_curve(test_label, predication, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        # Youden's J statistic to obtain the optimal probability threshold
        # best_threshold = sorted(list(zip(np.abs(tpr - fpr), predication)), key=lambda i: i[0], reverse=True)[0][1]
        best_threshold = 0.5
        print("threshold is {}".format(best_threshold))
        y_pred = [1 if i >= best_threshold else 0 for i in predication]
        confuMatrix = metrics.confusion_matrix(test_label, y_pred)
        confuMatrix = pd.DataFrame(confuMatrix, columns=['pred_0','pred_1'])
        print('Confusion Matrix(selected optimal threshold):')
        print(confuMatrix)
        plt.figure()
        if ROC == True:
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            plt.show()       
    
    def RandomForest(self, train, train_label, test, test_label = None, ROC = True):
        from sklearn.ensemble import RandomForestClassifier    
        clf = RandomForestClassifier(max_depth=5, criterion='gini')       
        clf.fit(train, train_label)    
        predication = clf.predict_proba(test)[:,1]
        if test_label is not None:
            self.CalResult(test_label,predication,ROC)
            return pd.DataFrame({"ID":test.index,"PROB":predication,"label":test_label})
        else:
            return pd.DataFrame({"ID":test.index,"PROB":predication})
        
    def HistGB(self, train, train_label, test, test_label = None, ROC = True):
        from sklearn.experimental import enable_hist_gradient_boosting
        from sklearn.ensemble import HistGradientBoostingClassifier        
        clf = HistGradientBoostingClassifier().fit(train, train_label)
        predication = clf.predict_proba(test)[:,1]
        if test_label is not None:
            self.CalResult(test_label,predication,ROC)
            return pd.DataFrame({"ID":test.index,"PROB":predication,"label":test_label})
        else:
            return pd.DataFrame({"ID":test.index,"PROB":predication})

def DataSplit(dataFrame,test_size = 0.33, seed = 0):
    from sklearn.model_selection import train_test_split
    data_train, data_test = train_test_split(dataFrame, test_size = test_size, random_state=seed,
                                             stratify = dataFrame['fraud'])
    test_label = data_test.loc[:,("claim_id","fraud","claim_amount")]
    test_label['claim_amount'] = test_label["claim_amount"].apply(lambda x: float(x.split(',')[0]) + \
                                                                                  float(x.split(',')[1])/100)
    test_label['fraud'] = test_label.fraud.map(dict(N=int(0), Y=int(1)))
    return data_train.copy(), data_test.drop(columns=['fraud','claim_amount']),test_label

def ModelApply(data_train, data_test, data_testLabel, model, validation):
    mymodel = MyModel()
    if validation == True and data_testLabel is not None:              
        if model == 'randomforest':           
            # validation
            prediction = mymodel.RandomForest(data_train.iloc[:,1:],data_train['fraud'],data_test,data_testLabel['fraud'])
            
        elif model == 'histGB':
            # validation
            prediction = mymodel.HistGB(data_train.iloc[:,1:],data_train['fraud'],data_test,data_testLabel['fraud'])    
            # # apply model to whole training set             
        else:
            print('PLEASE SPECIFY A VALID MODEL!!')
            
        return prediction
    if validation == False:
        if model == 'randomforest':  
            # apply model to whole training set
            final_pred = mymodel.RandomForest(data_train.iloc[:,1:],data_train['fraud'], data_test)
            final_pred.to_csv(model+"output.csv",index=False)
        elif model == 'histGB':
            final_pred = mymodel.HistGB(data_train.iloc[:,1:],data_train['fraud'], data_test)   
            final_pred.to_csv(model+"output.csv",index=False)
        else:
            print('PLEASE SPECIFY A VALID MODEL!!')
            
if __name__ == '__main__':    
    data_train_full = pd.read_csv("train.csv",sep=";")
    data_test_full = pd.read_csv("test.csv",sep=";")
    
    ################################
    ## PART1: Validation
    ################################
    data_subTrain, data_subTest, data_subTestLabel = DataSplit(data_train_full, seed = 20)
    data = DataPreprocess(data_subTrain, data_subTest)
    data.applyPreprocess()
    data.applyEncoder()
    data_subTrain = data.data_train
    data_subTest = data.data_test
    # # to solve unbalance of training set, under-sample data
    # data_subTrainsampled = data.UnderSampling(data_subTrain,4,5,seed=0) # randomness is controled by seed
    # alternatively over-sample
    data_subTrainsampled = data.UnderSampling(data_subTrain,5,4,seed=10) # randomness is controled by seed
    # fill missing value
    # data_subTrainsampled = data.FillNan(data_subTrainsampled)
    # data_subTest = data.FillNan(data_subTest)
    
    pred = ModelApply(data_subTrainsampled,data_subTest,data_subTestLabel,'histGB',validation = True)
    
    # compute top 100 sum of amount, divided by total fraud amount
    predTop100 = pred.sort_values(by=['PROB'],ascending=False)[0:100]
    predTop100 = predTop100.merge(data_subTestLabel,how='inner',left_on='ID', right_on='claim_id')
    predAmount = sum(predTop100['claim_amount'][predTop100['label']==1])/sum(data_subTestLabel['claim_amount'][data_subTestLabel['fraud']==1])
    print("TOP100 amount percentage is: {0:.3f}".format(predAmount))
    
    # ################################
    # ## PART2: Apply to whole training set
    # ################################
    # data = DataPreprocess(data_train_full, data_test_full)
    # data.applyPreprocess()
    # data.applyEncoder()
    # data_Train = data.data_train
    # data_Test = data.data_test
    # # to solve unbalance of training set, under-sample data
    # data_Trainsampled = data.UnderSampling(data_Train,5,4,seed=10) # randomness is controled by seed
    # # fill missing value
    # data_Trainsampled = data.FillNan(data_Trainsampled)
    # data_Test = data.FillNan(data_Test)   
    # # validation
    # ModelApply(data_Trainsampled, data_Test, data_testLabel = None, model = 'randomforest',validation = False)