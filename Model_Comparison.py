import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
import tensorflow
import tensorflow as tf
import datetime as dt
from tensorflow.keras.models import load_model
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot
import matplotlib
import matplotlib.pyplot as pyplot
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

## Loading the trained models that we wanna make comparison based on them
model_nest = load_model('resnest.h5')
model_net = load_model('resnet.h5')

#Defining Image sizes and channels
Image_Width=128
Image_Height=128
Image_Size=(Image_Width,Image_Height)
Image_Channels=3

## Reading and extracting Metadata
df = pd.read_csv("df.csv")
filename = df['filename']
category = df['category']

## Defining labels and train, validation, and test dataset
df["category"] = df["category"].replace({0:'real',1:'fake'})
train_df,testvalidate_df = train_test_split(df,test_size=0.30,
  random_state=12)
train_df = train_df.reset_index(drop=True)
testvalidate_df = testvalidate_df.reset_index(drop=True)
validate_df,test_df = train_test_split(testvalidate_df,test_size=0.50,
  random_state=12)
test_df = test_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
total_train=train_df.shape[0]
total_validate=validate_df.shape[0]
total_test=test_df.shape[0]
batch_size= 32


## Generating Training set, Validation Set, and Testing Set
train_datagen = ImageDataGenerator(rotation_range=15,
                                rescale=1./255,
                                shear_range=0.1,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                width_shift_range=0.1,
                                height_shift_range=0.1
                                )
train_generator = train_datagen.flow_from_dataframe(train_df,
                                                 "./Dataset/Dataset/",x_col='filename',y_col='category',
                                                 target_size=Image_Size,
                                                 class_mode='categorical',
                                                 batch_size=batch_size)
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    "./Dataset/Dataset/",
    x_col='filename',
    y_col='category',
    target_size=Image_Size,
    class_mode='categorical',
    batch_size=batch_size
)
test_datagen = ImageDataGenerator(rotation_range=15,
                                rescale=1./255,
                                shear_range=0.1,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                width_shift_range=0.1,
                                height_shift_range=0.1
                                )
test_generator = test_datagen.flow_from_dataframe(test_df,
                                                 "./Dataset/Dataset/",x_col='filename',y_col='category',
                                                 target_size=Image_Size,
                                                 class_mode='categorical',
                                                 batch_size=batch_size,
                                                 shuffle=False)

## Make Predictions based on trained models
nb_samples = test_df.shape[0]
predict_nest = model_nest.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size), verbose=1)
predict_net = model_net.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size), verbose=1)

## Calculating Accuracy
_, acc_nest = model_nest.evaluate_generator(test_generator,steps=np.ceil(nb_samples/batch_size), verbose=1)
print('ResNeSt Accuracy = > %.3f' % (acc_nest * 100.0))
_, acc_net = model_net.evaluate_generator(test_generator,steps=np.ceil(nb_samples/batch_size), verbose=1)
print('ResNet Accuracy = > %.3f' % (acc_net * 100.0))

## Plotting ROC Curve for both models
y_true = test_generator.classes
auc_nest = roc_auc_score(y_true, predict_nest[:,1])
print('ResNeSt ROC AUC=%.3f' % (auc_nest))
auc_net = roc_auc_score(y_true, predict_net[:,1])
print('ResNeSt ROC AUC=%.3f' % (auc_net))
pyplot.figure(figsize=(8,5))
fpr_nest, tpr_nest, _ = roc_curve(y_true, predict_nest[:,1])
fpr_net, tpr_net, _ = roc_curve(y_true, predict_net[:,1])
pyplot.plot(fpr_nest, tpr_nest, marker='.', label='ResNeSt')
pyplot.plot(fpr_net, tpr_net, marker='.', label='ResNet')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

## Precision Recall Curve
nest_y_pred = predict_nest > 0.5
nest_y_pred = list(map(int, nest_y_pred[:,1]))

net_y_pred = predict_net > 0.5
net_y_pred = list(map(int, net_y_pred[:,1]))

# precision-recall curve and f1
nest_precision, nest_recall, _ = precision_recall_curve(y_true,  predict_nest[:,1])
net_precision, net_recall, _ = precision_recall_curve(y_true,  predict_net[:,1])
nest_f1, nest_auc = f1_score(y_true,  nest_y_pred), auc(nest_recall, nest_precision)
net_f1, net_auc = f1_score(y_true,  net_y_pred), auc(net_recall, net_precision)
# summarize scores
print('ResNeSt: f1=%.3f auc=%.3f' % (nest_f1, nest_auc))
print('ResNet: f1=%.3f auc=%.3f' % (net_f1, net_auc))
# plot the precision-recall curves
pyplot.figure(figsize=(8,5))
pyplot.plot(nest_recall, nest_precision, marker='.', label='ResNeSt')
pyplot.plot(net_recall, net_precision, marker='.', label='ResNet')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()


