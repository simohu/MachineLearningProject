Machine Learning Project 2017/1028 - Simone Hu

- WHAT CONTAINS -
project.py contains all the code of my project

eurusd.ods is a file where I started to manipulate my data so is not important

data.csv contains all eur/usd’s values from 1989 to 2017 and is used by the program to create dataset.csv

Dataset.csv contains some random averages of different years of my data for the training set

test.csv contains the test set

In the end I exported my neural network in Model.h5 and model.json 



- WHAT IT DOES -
The purpose of the project is try to predict the FOREX’s markets, in particular EUR/USD using a avarage of N random years to predict the trend of the next year



— WHAT I INSTALLED —
In this project I installed Anaconda with Python 2.7, and I used Keras and Tensorflow libraries.



- HOW TO REPRODUCE -

To reproduce my experiment it is necessary execute the Python’s file project.py



- MY CODE -

First of all I red the data from data.csv and I created my trainingSet generating a lot of averages of N random years from 0 to M-1 (input of the traningset) and the year M (lablel of my traingset), then I saved all the data in dataset.csv because using a random calculations the dataset is always different, but I don’t want to change my dataset every time I execute my program so I saved it only for the first time.
After I scaled all the values of the dataset from 0 to 1, generated the neural network with Keras, compiled and fitted it, and in the end drawn the loss, accuracy charts, a random prediction of the trainingSet and a prediction of the testSet










