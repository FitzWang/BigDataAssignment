import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import backend as K
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
from tensorflow.keras.models import save_model, load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
pd.options.mode.chained_assignment = None


# input image and resize it
def image_input(path):
    path = "recipes/"+str(path)+".png"
    img = load_img(path, target_size=(224, 224))
    img = img_to_array(img)
    # img = preprocess_input(img)
    return img


# Weighted cross-entropy loss function

def mycrossentropy(y_true, y_pred):
    weights = 0.07
    # need to set the weight by yourself
    # weights stands for how much you emphasize on correct prediction for 1
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.zeros_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.ones_like(y_pred))
    loss = (1-weights)*K.binary_crossentropy(y_true, y_pred)*pt_1+weights*K.binary_crossentropy(y_true, y_pred)*pt_0
    return loss




# model setting
def baseline_model():
    # create model
    pretrained_model = MobileNetV2(weights="imagenet", include_top=False,  input_shape=(224, 224, 3))
    for layer in pretrained_model.layers:
        layer.trainable = False

    model = Sequential()
    model.add(pretrained_model)
    model.add(Flatten())
    model.add(Dense(97, activation='sigmoid'))

    # Compile model
    model.compile(loss=mycrossentropy, optimizer=Adam(learning_rate=0.0001), metrics=['AUC'])
    return model

# model training
def model_train(train_x, train_y):
    model = baseline_model()
    # train_y is a data frame
    train_y =  np.asarray(train_y)
    cum = 0
    batch_x = []
    batch_y = []
    for i in range(train_x.shape[0]):
        cum+=1
        img = image_input(train_x.iloc[i][0])
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        target = train_y[i, :]
        target = np.expand_dims(target, axis=0)

        # the original data set is too large, each time we read 300 samples
        if len(batch_x) == 0:
            batch_x = img
            batch_y = target
        else:
            batch_x = np.vstack([batch_x, img])
            batch_y = np.vstack([batch_y, target])

        # after reading 300 sample, we train the model
        if cum == 300 or i == train_x.shape[0]-1:
            # problems that need to be fix for multi-label

            # weights = class_weight.compute_class_weight('balanced',
            #                                             np.unique(batch_y),
            #                                             batch_y)

            # type of batch_x and batch_y should be the same
            batch_y = tf.cast(batch_y, tf.float32)
            # update the weights every 30 samples and every input is used for 3 times
            model.fit(batch_x, batch_y,  batch_size=30, epochs=3, verbose=2)
            # reset the counting and input
            cum = 0
            batch_x = []
            batch_y = []

    return model


def evaluate(test_x, test_y, model):
    test_y = np.asarray(test_y)

    total = test_y.shape[1]*test_x.shape[0]
    cum_tp_tn = cum_tp_fp = cum_tp_fn = cum_true_pos = 0
    test_status = np.empty((0, 3))

    for i in range(test_x.shape[0]):

        img = image_input(test_x.iloc[i][0])
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        target = test_y[i, :]
        target = np.expand_dims(target, axis=0)

        # predicted label
        yhat = model.predict(img)
        yhat = yhat.round()

        # build confusion matrix
        cm = confusion_matrix(target[0], yhat[0])
        if len(cm[0]) > 1:
            true_pos = cm[1][1]
            tp_tn = np.sum(np.diag(cm))
            tp_fp = np.sum(cm[0])
            tp_fn = cm[0][0] + cm[1][0]
        else:
            true_pos = 0
            tp_tn = np.sum(np.diag(cm))
            tp_fp = np.sum(cm[0])
            tp_fn = cm[0][0]

        cum_true_pos += true_pos
        cum_tp_tn += tp_tn
        cum_tp_fp += tp_fp
        cum_tp_fn += tp_fn

        accuracy = tp_tn/np.sum(np.sum(cm))
        precision = true_pos / tp_fp
        recall = true_pos/ tp_fn
        test_status = np.vstack([test_status, [accuracy, precision, recall]])

    status = {"Accuracy": [cum_tp_tn/total], "precision": [cum_true_pos/ cum_tp_fp], "recall": [cum_true_pos/cum_tp_fn]}
    status = pd.DataFrame(status)

    return test_status, status


# read data
recipes= pd.read_csv("recipes.csv", sep=';')
# remove one damaged figure
recipes.drop(recipes[recipes["photo_id"] == "CMmV2A2sBvY"].index, inplace=True)
recipes_clean = recipes.iloc[:, np.r_[0, 7:recipes.shape[1]]]

# change the string to list
ima_list = [0]*recipes_clean.shape[0]
for i in range(recipes_clean.shape[0]):
    ima_list[i] = [recipes_clean.iloc[i, 0]]
recipes_clean["photo_id"] = ima_list

# split the data set to train and test
train, test = train_test_split(recipes_clean, test_size=0.3, train_size=0.7,
                               random_state=8, shuffle=True)
# train, validation = train_test_split(train, test_size=0.5, train_size=0.5,
#                                 random_state=8, shuffle=True)

# further split the label and input
train_x, train_y = train.iloc[:, 0], train.iloc[:, np.r_[1:train.shape[1]]]
# validation_x, validation_y = validation.iloc[:, 0], validation.iloc[:, np.r_[1:validation.shape[1]]]
test_x, test_y = test.iloc[:, 0], test.iloc[:, np.r_[1:test.shape[1]]]



model = model_train(train_x, train_y)
# model = baseline_model()
# save the trained model
save_model(model, "MobileNetV2_model.h5")
model = load_model("MobileNetV2_model.h5", compile=False)

test_status, status = evaluate(test_x, test_y, model)

np.savetxt("test_status.csv", test_status, delimiter=",")
status.to_csv("status.csv")