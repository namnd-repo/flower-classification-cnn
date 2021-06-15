# Flower Classification
Simple Flower Classification with CNN using Keras.

<br>

## Built with
[Pycharm](https://www.jetbrains.com/pycharm/download/)

## Dataset Description
The dataset consists of 3670 images of flowers divided into 5 different categories.

Download link: https://www.tensorflow.org/datasets/catalog/tf_flowers.

## Installing
* For installing, download ZIP or clone project:
<pre>
git clone https://github.com/namnd-repo/flower-classification-cnn.git
</pre>

* Create new project on Pycharm

Location is the downloaded "flower-classification-cnn" > Create from existing sources.

* Install environment

File > Settings > Project > Project Interpreter.

Install Keras, cv2, Numpy, Sklearn, Matplotlib. 

* Dataset

Copy downloaded dataset to the project (some folders need to be renamed). Following the folder structure:

![image](https://user-images.githubusercontent.com/85830956/121950367-bec62a80-cd83-11eb-9dde-0c7811da2804.png)

## Running the program
preprocess.py >> train.py >> predict.py

After running preprocess.py and train.py, "data.pickle" and "model.h5" will be created.

## Results
Accuracy: **92.10**%
