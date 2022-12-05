# Deep Fake Face Recognition using ResNeSt
Birkbeck University - Master's Degree Project - Deep Fake Facial image Recognition using ResNest\
Please read the "README" file for further information and instructions on how to run the codes.\
.\
.\
.\
Can you decide which of these faces are "REAL" or "FAKE"?!\
![DeepFAKE](/deepfake[1].png)\
This project aims to detect these fake facial images using deep learning methods with the acuracy of 97 percent.\
.\
.\
Answer: The left-hand side image is "FAKE" and the right-hand side image is "REAL"!
# Instructions on How to Run the Project
Please follow the instructions as below:
## 1- Download the Required Files to the Main Directory
Please download the required files using the DropBox link below:\
https://www.dropbox.com/sh/3lxarwtc7kzeref/AAAdBxZXDOJ2jFd11qjKjHXca?dl=0 \
There are four files in this directory. \
"Dataset.zip" - This is the raw dataset used for this project. that contains all images.\
"df.csv" - This CSV file contains metadata, that is the label related to each filename in the dataset.\
"resnest.h5" - This is the trained model of "ResNeSt" for making predictions, or running the GPU.\
"resnet.h5" - This is the trained model of "ResNet" for making predictions, or running the GPU.\
## 2- Unzip the "Dataset.zip" File
Please make a folder named "Dataset" in the main directory.\
Please unzip the "Dataset.zip" file, using "unzip.py" code.\
This code will automatically unzip the file and save it as "Dataset" to the main directory, which is the name of our dataset directory in codings.
## 3- Metadata
The "df.csv" file contains metadata. Which includes the filenames along with their "Fake" or "Real" label. Please make sure that this file is downloaded into your working directory as it is needed for all the project codes.
## 4- ResNeSt
In order to run the "ResNeSt" model training procedure, please run "resnest.py" after making sure all the pre-requirements are met and all the required files mentioned in 1,2 and 3 are there in your main directory. Running this file will do the following tasks: pre-process the data, make ResNeSt model, train the model, apply early stopping, making predictions and showing accuracy on test dataset, plotting training history, plotting ROC, calculating AUC,  and P-R Curves, and plotting Confusion Matrix.
## 5- ResNet
In order to run the "ResNet" model training procedure, please run "resnet.py" after making sure all the pre-requirements are met and all the required files mentioned in 1,2 and 3 are there in your main directory. Running this file will do the following tasks: pre-process the data, make ResNet model, train the model, apply early stopping, making predictions and showing accuracy on test dataset, plotting training history, plotting ROC, calculating AUC, and plotting Confusion Matrix.
## 6- ResNet and ResNeSt Comparison and Obtaining Curves
In order to obtain the ROC and P-R curves for model comparison, please run the "Model_Comparison.py". Running this file does the following tasks: will preprocess image data, makes train, test, validation set, loads both resnet and resnest models, makes prediction based on two models, plots ROC curve for two models, plots P-R curves for both models.
## 7- Running GUI
To run the GUI you need to have the trained model file saved in the related directory of "GUI.py". Therefore, make sure trained models obtained from training or downloaded trained models from the DropBox are there in the directory. Afterwards, make sure that "Tkinter" is installed on Python. Finally, run the "GUI.py" file to make the GUI appear on your screen. The GUI is completely user-friendly, so all you need is to upload a random image and then press "classify image" button to see the results.  
## 8- prerequisites:
In order to make these codes work, you need to install these packages below. Please make sure all these packages are installed in their latest version, as the codes are based on the latest versions.\
.Tkinter\
.PIL\
.numpy\
.Scikit-learn\
.tensorfelow\
.keras\
.pandas\
.matplotlib\
.os\
.datetime\
.time
