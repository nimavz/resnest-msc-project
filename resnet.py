"https://arxiv.org/pdf/1512.03385.pdf"
"https://data-flair.training/blogs/cats-dogs-classification-deep-learning-project-beginners/"

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
import tensorflow
import datetime as dt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import seaborn as sn
import pandas as pd
from matplotlib import pyplot
import sys
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

## Initialising image size and channels
Image_Width=128
Image_Height=128
Image_Size=(Image_Width,Image_Height)
Image_Channels=3


## Setting up the directory and labels. Please note this directory might differ
## So, please adjust directory based on your computer.

## This part of code is not needed of you are using metadata
# dir = "./Dataset/Dataset"
# labels = ['fake', 'real']
# categories=[]
# filename = []
# for label in labels:
#     for f_name in os.listdir(dir + '/' + label + '/'):
#         category = label
#         filename.append(f_name)
#         if category=='fake':
#             categories.append(1)
#         else:
#             categories.append(0)
# df=pd.DataFrame({
#     'filename':filename,
#     'category':categories
# })

## Reading and extracting Metadata
df = pd.read_csv("df.csv")
filename = df['filename']
category = df['category']

##Converting Labels and Defining Batch Size
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

## Defining Early Stopping Conditions
earlystop = EarlyStopping(patience = 50)
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc',patience = 50,verbose = 1,factor = 0.5,min_lr = 0.01)
callbacks = [earlystop,learning_rate_reduction]


#### RESNET MODEL #####

## Defining Functions
def res_net_block(input_data, filters, conv_size):
  x = tensorflow.keras.layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
  x = tensorflow.keras.layers.BatchNormalization(momentum=0.01)(x)
  x = tensorflow.keras.layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
  x = tensorflow.keras.layers.BatchNormalization(momentum=0.01)(x)
  x = tensorflow.keras.layers.Add()([x, input_data])
  x = tensorflow.keras.layers.Activation('relu')(x)
  return x


callbacks = [
  # Write TensorBoard logs to `./logs` directory
  tensorflow.keras.callbacks.TensorBoard(log_dir='./log/{}'.format(dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), write_images=True),
]


inputs = tensorflow.keras.Input(shape=(128, 128, 3))
x = tensorflow.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
x = tensorflow.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = tensorflow.keras.layers.MaxPooling2D(3)(x)

num_res_net_blocks = 4
for i in range(num_res_net_blocks):
  x = res_net_block(x, 64, 3)

x = tensorflow.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = tensorflow.keras.layers.GlobalAveragePooling2D()(x)
x = tensorflow.keras.layers.Dense(256, activation='relu')(x)
x = tensorflow.keras.layers.Dropout(0.5)(x)
outputs = tensorflow.keras.layers.Dense(2, activation='softmax')(x)

res_net_model = tensorflow.keras.Model(inputs, outputs)

## Compile the model
res_net_model.compile(optimizer=tensorflow.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['acc'])

## Training the model
epochs=100
t1 = time.time()
history = res_net_model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks, verbose=1)
tot_time = time.time() - t1
print('Trainig took %s seconds' %round(tot_time))

## Saving the model
res_net_model.save("resnet.h5")


## Plotting the training Procedure
def summarize_diagnostics(history):
    # plot loss
    pyplot.figure(figsize=(10,8))
    pyplot.subplot(212)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='val')
    pyplot.legend()
    # plot accuracy
    pyplot.subplot(211)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['acc'], color='blue', label='train')
    pyplot.plot(history.history['val_acc'], color='orange', label='val')
    pyplot.legend()
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.49)
    pyplot.show(filename)
    pyplot.close()

summarize_diagnostics(history)

## Maked predictions based on testing dataset
nb_samples = test_df.shape[0]
predict = res_net_model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))

## Accuracy Calculations
_, acc = res_net_model.evaluate_generator(test_generator,steps=np.ceil(nb_samples/batch_size), verbose=1)
print('> %.3f' % (acc * 100.0))


## Confusion Matrix
y_true = test_generator.classes
y_pred = predict > 0.5
y_pred = list(map(int, y_pred[:,1]))
font = {
'family': 'Times New Roman',
'size': 12
}
matplotlib.rc('font', **font)
mat = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(mat, range(2), range(2))
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16},fmt='g') # font size

plt.show()


## Plotting ROC Curve and Calculating AU-ROC for RESNET
auc = roc_auc_score(y_true, predict[:,1])
print('ResNeSt ROC AUC=%.3f' % (auc))
fpr, tpr, _ = roc_curve(y_true, predict[:,1])
pyplot.plot(fpr, tpr, marker='.', label='ResNeSt')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
