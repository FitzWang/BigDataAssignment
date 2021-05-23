### at least three places need to be change
### change the name here !!!
model_name = 'Adam_-1_'
import tensorflow as tf
from keras import optimizers
import matplotlib
import os
from keras.models import load_model
import numpy as np
from PIL import Image
import cv2
from keras import backend as K
import json

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

### change your model path here 
path = "E:/Code/KUL/big_data_world/second/{}.h5".format(model_name)

model = load_model(path, compile=False)
model.compile(loss='binary_crossentropy',
                optimizer=optimizers.Adam(lr=1e-4),
                metrics=[f1,tf.keras.metrics.AUC(),precision,recall,'acc'])

# model.summary()
def get_inputs(src=[]):
    pre_x = []
    for s in src:
        input = cv2.imread(s)
        input = cv2.resize(input, (img_size, img_size))
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        pre_x.append(input)  # input一张图片
    pre_x = np.array(pre_x) / 255.0
    return pre_x

### test pictures stored in one folder here, form them in .jpg
### change your path here !!!
predict_dir = 'C:/Users/Think/Desktop/predict'
### your img size for model
img_size = 224
### the record_json path here
### change your path here !!!
out_json_path = 'E:/Code/KUL/big_data_world/second/record_dict.json'
with open(out_json_path,"r") as load_f:
    record_dict = json.load(load_f)
features_list = list(record_dict[model_name]['record_feature_indicators'].keys())
for fn in os.listdir(predict_dir):
    if fn.endswith('jpg'):
        images = []
        fd = os.path.join(predict_dir, fn)
        images.append(fd)
        pre_x = get_inputs(images)
        pre_y = model.predict(pre_x)

        for pre_ in pre_y:
            dict_ = {}
            for i in range(len(pre_)):
                dict_[features_list[i]] = pre_[i]
            dict_ = sorted(dict_.items(), key = lambda kv:(kv[1], kv[0]),reverse = True)
            print (fn)
            print (dict_)
