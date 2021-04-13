import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img
### change to your own model
### other things to find out: 1.optimizers; 2.frozen layers num; 3.DataAugmentation; 4. NN building
from keras.applications import NasNetLarge
from keras import models
from keras import layers
from keras import optimizers
import io
from PIL import Image
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import roc_auc_score,f1_score,precision_score,recall_score,accuracy_score


def get_weight(weights=0.07):
    # need to set the weight by yourself
    # weights stands for how much you emphasize on correct prediction for 1
    def mycrossentropy(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.zeros_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.ones_like(y_pred))
        loss = (1-weights)*K.binary_crossentropy(y_true, y_pred)*pt_1+weights*K.binary_crossentropy(y_true, y_pred)*pt_0
        return loss

    return mycrossentropy

def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        y_true = tf.cast(y_true, tf.float32)
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall_ = true_positives / (possible_positives + K.epsilon())
        return recall_

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    y_true = tf.cast(y_true, tf.float32)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_ = true_positives / (predicted_positives + K.epsilon())
    return precision_

def f1(y_true, y_pred):
    precision_ = precision(y_true, y_pred)
    recall_ = recall(y_true, y_pred)
    return 2*((precision_*recall_)/(precision_+recall_+K.epsilon()))


def ModeChange(pretrained_model,cut_idx):
    for layer in pretrained_model.layers[:cut_idx]:
        layer.trainable = False
    return pretrained_model

def TransferLearning(pretrained_model,cut_idx,DataMode,record_dict):
    
    
    pretrained_model = ModeChange(pretrained_model,cut_idx)
    
    # for layer in pretrained_model.layers:
    #     print(layer, layer.trainable)

    '''
    change your NN here
    '''

    model = models.Sequential()
    model.add(pretrained_model)
    model.add(layers.Flatten())
    ### how many neurons for he hidden layer before full-connected layer
    model.add(layers.Dense(512, activation='relu'))
    ### the ratio of dropout
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    ### the ratio of dropout
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(len(test_feature), activation='sigmoid'))
    ## loss='binary_crossentropy',
    ## loss=get_weight(),
    model.compile(loss='binary_crossentropy',
                optimizer=chosen_optimizer,
                metrics=[f1,tf.keras.metrics.AUC(),precision,recall,'acc'])

    ### 3. DataAugmentation
    if DataMode == 'DataAugmentation':
        train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
        validation_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)
        validation_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_batchsize = batchsize_
    val_batchsize = batchsize_
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=TrainDf,
        directory=rootDir ,
        x_col="photo_id",
        y_col=test_feature,
        # subset="training",
    #   classes=labels,
        target_size=[image_size, image_size],
        batch_size=train_batchsize,
        class_mode='raw'
    )
    ## multi_output  raw
    # labels = (train_generator.class_indices)
    # print (labels)
    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=ValDf,
        directory=rootDir ,
        x_col="photo_id",
        y_col=test_feature,
        # subset="validation",
    #   classes=labels,
        target_size=[image_size, image_size],
        batch_size=train_batchsize,
        class_mode='raw'
    )
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=TestDf,
        directory=rootDir ,
        x_col="photo_id",
        # y_col=test_feature,
        # subset="validation",
        class_mode=None,  
        target_size=[image_size, image_size],
        batch_size=1,
        # class_mode='raw'
    )

    
    if DataMode == 'DataAugmentation':
        steps_per_epoch = 2*train_generator.samples/train_generator.batch_size
        epochs=2*train_epochs
    else:
        steps_per_epoch = train_generator.samples/train_generator.batch_size
        epochs=train_epochs
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples/validation_generator.batch_size,
        verbose=1
    )

    test_generator.reset()
    pred=model.predict_generator(
        test_generator,
        steps=test_generator.samples/test_generator.batch_size,
        verbose=1)

    

    print (history.history)
    save_pref = '_'.join([optimizer,str(cut_idx),DataMode])
    model.save('{}.h5'.format(save_pref))
    record_dict[save_pref] = {}
    record_dict[save_pref] = history.history
    
    pred_bool = (pred >=0.5)
    predictions = pred_bool.astype(int)
    train_f1 = history.history['f1']
    val_f1 = history.history['val_f1']
    record_feature_indicators = {}
    for idx in range(len(TestLabel[0])):
        SubTestLabel = [i[idx] for i in TestLabel]
        SubPredictions = [i[idx] for i in predictions]
        SubPredProb = [i[idx] for i in pred]
        sub_roc = roc_auc_score(y_true=SubTestLabel,y_score=SubPredProb)
        sub_f1 = f1_score(y_true=SubTestLabel,y_pred=SubPredictions)
        sub_precision = precision_score(y_true=SubTestLabel,y_pred=SubPredictions)
        sub_recall = recall_score(y_true=SubTestLabel,y_pred=SubPredictions)
        sub_acc = accuracy_score(y_true=SubTestLabel,y_pred=SubPredictions)
        record_feature_indicators[test_feature[idx]] = {}
        record_feature_indicators[test_feature[idx]]['sub_roc'] = sub_roc
        record_feature_indicators[test_feature[idx]]['sub_acc'] = sub_acc
        record_feature_indicators[test_feature[idx]]['sub_precision'] = sub_precision
        record_feature_indicators[test_feature[idx]]['sub_recall'] = sub_recall
        record_feature_indicators[test_feature[idx]]['sub_f1'] = sub_f1
    # record_feature_indicators = sorted(record_feature_indicators.items(), key = lambda kv:(kv[1], kv[0]),reverse = True)
    print (record_feature_indicators)    
    record_dict[save_pref]['record_feature_indicators'] = record_feature_indicators    

    test_acc = round(accuracy_score(y_true=TestLabel,y_pred=predictions),6)
    record_dict[save_pref]['test_acc'] = test_acc
    test_auc = round(roc_auc_score(y_true=TestLabel,y_score=pred),6)
    record_dict[save_pref]['test_auc'] = test_auc
    micro_test_f1 = round(f1_score(y_true=TestLabel,y_pred=predictions, average='micro'),6)
    macro_test_f1 = round(f1_score(y_true=TestLabel,y_pred=predictions, average='macro'),6)
    record_dict[save_pref]['micro_test_f1'] = micro_test_f1
    record_dict[save_pref]['macro_test_f1'] = macro_test_f1
    test_precision = np.float(round(precision(y_true=TestLabel,y_pred=pred).numpy(),6))
    test_recall = np.float(round(recall(y_true=TestLabel,y_pred=pred).numpy(),6))
    record_dict[save_pref]['test_precision'] = test_precision
    record_dict[save_pref]['test_recall'] = test_recall

    try:
        train_auc = history.history['auc']
        val_auc = history.history['val_auc']
    except:
        for key_ in history.history.keys():
            if key_.startswith('auc'):
                train_auc = history.history[key_]
            elif key_.startswith('val_auc'):
                val_auc = history.history[key_]
    

    # history = model.fit_generator(
    #     test_generator,
    #     steps_per_epoch=steps_per_epoch,
    #     epochs=epochs,
    #     verbose=1
    # )
    # test_acc = history.history['acc']
    # test_loss = history.history['loss']
    
    epochs = range(len(train_f1))
    plt.figure()
    plt.plot(epochs, train_f1, 'b', label='Traning F1')
    plt.plot(epochs, val_f1, 'r', label='Validation F1')
   # # plt.plot(epochs, test_f1, 'g', label='Test F1')
    plt.title('Training and Validation F1')
    plt.legend()
    plt.savefig("./{}_F1.png".format(save_pref))
    
    plt.figure()
    plt.plot(epochs, train_auc, 'b', label='Traning AUC')
    plt.plot(epochs, val_auc, 'r', label='Validation AUC')
    ## plt.plot(epochs, test_auc, 'g', label='Test AUC')
    plt.title('Training and Validation AUC')
    plt.legend()
    plt.savefig("./{}_AUC.png".format(save_pref))
#     plt.show()

    print ('Micro Test F1:{}, Macro Test F1:{} and AUC:{}'.format(str(micro_test_f1),str(macro_test_f1),str(test_auc)))

    return record_dict

def MergeLables(df,merge_threshold):
    total_record_list = []
    test_feature = [i for i in df.columns.tolist() if i.startswith('tag_')]
    print (len(test_feature))
    print (test_feature[:10])
    for idx_,feature_ in enumerate(test_feature):
        original_feature_list = list(df[feature_])
        for idx_2,feature_2 in enumerate(test_feature[idx_+1:]):
            compare_feature_list = list(df[feature_2])
            sum_feature_list = list(np.sum([original_feature_list, compare_feature_list], axis = 0))
            union_len = len([i for i in sum_feature_list if i>=1])
            intersection_len = len([i for i in sum_feature_list if i==2])
            merge_ration = intersection_len/union_len
            total_record_list.append([feature_,feature_2,merge_ration])
    
    total_record_list = sorted(total_record_list,key=lambda x: x[2],reverse = True)
    print (total_record_list[:10])
    if total_record_list[0][2] >= merge_threshold:
        print (total_record_list[0])
        new_binary_list = [1 if list(df[total_record_list[0][0]])[i] == 1 or list(df[total_record_list[0][1]])[i] == 1 else 0  for i in range(df.shape[0])]
        df['_'.join([total_record_list[0][0],total_record_list[0][1]])] = new_binary_list
        df = df.drop([total_record_list[0][0],total_record_list[0][1]],1)
        df = MergeLables(df,merge_threshold)
        return df
    else:
        return df

# def PreprocessingSamples(dataframe_,dataframe_offset,dataframe_offset_2):
#     dataframe_take = dataframe_.copy()
#     test_feature = [i for i in dataframe_.columns.tolist() if i.startswith('tag_')]
#     dataframe_2 = dataframe_[test_feature]
#     count_ = 0
#     for idx_ in range(dataframe_2.shape[0]):
#         sample = [i for i in dataframe_2.iloc[idx_,:]]
#         if sum(sample) == 0:
#             count_ += 1
#             # print (dataframe_.shape[0],dataframe_offset.shape[0],dataframe_offset_2.shape[0])
#             # prob_ = random.random()
#             # if prob_ <= 0.34:
#             #     dataframe_offset = dataframe_offset.append(dataframe_take.iloc[idx_,:])
#             #     dataframe_ = dataframe_.drop(idx_,0)
#             # elif 0.34 < prob_ <= 0.67:
#             #     dataframe_offset_2 = dataframe_offset_2.append(dataframe_take.iloc[idx_,:])
#             dataframe_ = dataframe_.drop(idx_,0)

#     print (count_)
#     return dataframe_,dataframe_offset,dataframe_offset_2

def PreprocessingSamples(dataframe_):
    test_feature = [i for i in dataframe_.columns.tolist() if i.startswith('tag_')]
    dataframe_2 = dataframe_[test_feature]
    count_ = 0
    for idx_ in range(dataframe_2.shape[0]):
        sample = [i for i in dataframe_2.iloc[idx_,:]]
        if sum(sample) == 0:
            count_ += 1
            # print (dataframe_.shape[0],dataframe_offset.shape[0],dataframe_offset_2.shape[0])
            # prob_ = random.random()
            # if prob_ <= 0.34:
            #     dataframe_offset = dataframe_offset.append(dataframe_take.iloc[idx_,:])
            #     dataframe_ = dataframe_.drop(idx_,0)
            # elif 0.34 < prob_ <= 0.67:
            #     dataframe_offset_2 = dataframe_offset_2.append(dataframe_take.iloc[idx_,:])
            dataframe_ = dataframe_.drop(idx_,0)

    print (count_)
    return dataframe_

def PreprocessingFeatures(dataframe_):
    test_feature = [i for i in dataframe_.columns.tolist() if i.startswith('tag_')]
    dataframe_2 = dataframe_[test_feature]

    portion_list = []
    portion_dict = {}
    for idx_,feature_ in enumerate(test_feature):
        original_feature_list = list(dataframe_[feature_])
        portion = round(sum(original_feature_list) / len(original_feature_list),4)
        portion_dict[feature_] = portion
        portion_list.append(portion)

        if portion < 0.01:
            print (feature_,' dropping...')
            dataframe_ = dataframe_.drop(feature_,1)

    portion_dict = sorted(portion_dict.items(), key = lambda kv:(kv[1], kv[0]),reverse = True)
    print (portion_dict)

    return dataframe_

def MergeLablesMannual(dataframe_):
    merge_dict = {'tag_ketodiet':['tag_keto', 'tag_ketodiet','tag_lowcarb','tag_diet', 'tag_weightloss','tag_fitness'],
    'tag_breakfasts':['tag_breakfastideas','tag_breakfast','tag_oats', 'tag_oatmeal'],
    'tag_italianstyle':['tag_italian', 'tag_italy', 'tag_italia','tag_italianfood','tag_pasta', 'tag_pastalover'],
    'tag_vegetable':['tag_veggie','tag_vegetarianrecipes', 'tag_vegetables', 'tag_veggies','tag_vegan', 'tag_veganfood','tag_plantbased','tag_vegetarian','tag_veganrecipe'],
    'tag_meals':['tag_brunch','tag_dinner', 'tag_lunch'],
    'tag_desserts':['tag_chocolat','tag_icecream','tag_cookies','tag_cakes','tag_sweettooth','tag_baker','tag_dessert', 'tag_desserts','tag_sweet', 'tag_sweets','tag_baking', 'tag_cake','tag_chocolate','tag_pastry','tag_bake', 'tag_bakery','tag_pancakes','tag_cheesecake','tag_bread'],
    'tag_fish':['tag_fish', 'tag_salmon','tag_seafood'],
    'tag_meategg':['tag_bbq','tag_protein','tag_eggs', 'tag_egg','tag_chicken', 'tag_steak','tag_meat', 'tag_beef'],
    'tag_organic': ['tag_organic', 'tag_natural'],
    'tag_fruit': ['tag_lemon','tag_fruit','tag_banana','tag_strawberry', 'tag_avocado'],
    'tag_asianstyle': ['tag_noodles','tag_rice','tag_curry', 'tag_indianfood','tag_asianfood'],
    'tag_snacks': ['tag_snack', 'tag_snacks'],
    'tag_webs':['tag_foodbloggers','tag_food52','tag_easyrecipes','tag_ilovefood'],
    'tag_drink':['tag_drinks','tag_coffee', 'tag_smoothie']}

    for key_ in merge_dict.keys():
        sub_dataframe_ = dataframe_[merge_dict[key_]]
        new_list = []
        for index_ in range(sub_dataframe_.shape[0]):
            new_list.append(max(sub_dataframe_.iloc[index_,:]))
        dataframe_[key_] = new_list
        dataframe_ = dataframe_.drop(merge_dict[key_],1)
    return dataframe_

def ClearNonValidPhotos(df_):
    photos = list(df_["photo_id"])
    for photo in photos:
        img = tf.io.gfile.GFile(os.path.join(rootDir,photo),'rb')
        try:
            encoded_jpg = img.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = Image.open(encoded_jpg_io)
            height,width = image.size
        except:
            # print (photo)
            df_ = df_[~df_['photo_id'].isin([photo])] 
    return df_


if __name__ == '__main__':

    '''
    your pictures folder dir path, must changed here!!! you could have a try in new folder with much less pictures
    '''
    rootDir = '/Users/lunachang/OneDrive - KU Leuven/Y1_S2/🔴 Advanced Analytics in Big Data World/Assignment/2_deep learning on images/recipes'
    df_path = 'recipes.csv'
    dataframe_ = pd.read_csv(df_path,sep = ';')
    ### the performance indicators will be saved as this name
    out_json_path = './record_dict.json'

    if os.path.exists(out_json_path):
        with open(out_json_path,"r") as load_f:
            record_dict = json.load(load_f)
            print("loading...")
    else:
        record_dict = {}
        print ('creating...')

    dataframe_ = dataframe_.dropna(axis=0,how='any').reset_index(drop=True)

    dataframe_ = PreprocessingSamples(dataframe_)

    dataframe_ = MergeLablesMannual(dataframe_)


    ### the auto merging method
    # merge_threshold = 0.3
    # dataframe_ = MergeLables(dataframe_,merge_threshold)

    dataframe_ = PreprocessingFeatures(dataframe_)
    print (dataframe_.shape[0],dataframe_.shape[1])

    test_feature = [i for i in dataframe_.columns.tolist() if i.startswith('tag_')]

    extract_feature = ['photo_id'] + test_feature[:]
    sub_dataframe_ = dataframe_[extract_feature]
    # sub_dataframe_[test_feature] = sub_dataframe_[test_feature].astype(str)  
    def append_ext(fn):
        return fn+".png"
    sub_dataframe_["photo_id"]=sub_dataframe_["photo_id"].apply(append_ext)

    ### Clear those data of pictures could not be found or loaded
    sub_dataframe_ = ClearNonValidPhotos(sub_dataframe_)
    print (sub_dataframe_.shape[0],sub_dataframe_.shape[1])

    ###y our portion of val and test portion, must changed!!!
    test_portion = 0.2
    validation_portion = 0.3
    TestDf = sub_dataframe_.sample(frac=test_portion)
    TrainDf = sub_dataframe_.drop(list(TestDf.index))
    ValDf = TrainDf.sample(frac=validation_portion)
    TrainDf = TrainDf.drop(list(ValDf.index))
    # TrainDf,ValDf,TestDf = PreprocessingSamples(TrainDf.reset_index(drop=True),ValDf.reset_index(drop=True),TestDf.reset_index(drop=True))
    
    TestLabel = TestDf[test_feature].values.tolist()
    
    ### the size of image, it could change for different model
    image_size = 331
    ### batch_size, you could reduce if your machine could not afford (exceeds free system memory)
    batchsize_ = 100
    ### how many train epoches, must changed here!!!
    train_epochs = 10
    optimizer = 'Adam'
    ### 1.optimization it is said that we should not use large learning rate for fine tune to avoid overfitting, so SGD is better.
    # optimizers.SGD(lr=1e-4)
    ## Adam  RMSprop
    if optimizer == 'SGD':
        chosen_optimizer = optimizers.SGD(lr=1e-4)
    elif optimizer == 'Adam':
        chosen_optimizer = optimizers.Adam(lr=1e-4)
    ### 0.model
    '''
    your model, must changed here! you could find here: https://keras.io/api/applications/
    Dont forget to import it above
    '''
    pretrained_model = NASNetLarge(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    
    ##  frozen all layers 
    record_dict = TransferLearning(pretrained_model,-1,'',record_dict)
    print (record_dict)
    out_json_dict = json.dumps(record_dict,indent=4)
    f3 = open(out_json_path, 'w')
    f3.write(out_json_dict)
    f3.close()
    ### 2. frozen layers num, you could change here to see the performance, or add new like:
    '''
    record_dict = TransferLearning(pretrained_model,the reverse index you want,'',record_dict)

    out_json_dict = json.dumps(record_dict,indent=4)
    f3 = open(out_json_path, 'w')
    f3.write(out_json_dict)
    f3.close()
    '''
    ###  but be carful about overfitting!
    ## 
    record_dict = TransferLearning(pretrained_model,-2,'',record_dict)

    out_json_dict = json.dumps(record_dict,indent=4)
    f3 = open(out_json_path, 'w')
    f3.write(out_json_dict)
    f3.close()

    ## 3. with DataAugmentation
    record_dict = TransferLearning(pretrained_model,-3,'',record_dict)

    out_json_dict = json.dumps(record_dict,indent=4)
    f3 = open(out_json_path, 'w')
    f3.write(out_json_dict)
    f3.close()

    



