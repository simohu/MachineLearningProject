import numpy as np
import pandas as pd
from random import randint
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Read data
data = pd.read_csv("./data.csv", engine='python')

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

series = data.values
rows, columns = np.shape(series)
m = []

for i in range(0,rows):
    if (i > 9):
        for j in range(0, i*3):
            row = randint(1,i)
            sumseries = series[i-1]  
            #print sumseries
            for k in range(0, row-1):
                randrows = range(0, i-1)
                r = randint(0,len(randrows)-1)
                r = randrows[r]
                randrows.remove(r)
                sumseries = sumseries + series[r]
            avarageseries = sumseries/row
            m.append(np.concatenate((avarageseries,series[i]), axis=0))

#numpy.savetxt("dataset.csv", m, delimiter=",")
data = pd.read_csv("./dataset.csv", engine='python')
series = data.values

# Specify the data 
X=series[:,0:260]

# 0 to 1
y=series[:,260:520]
y = np.transpose(y)
X = np.transpose(X)
for i in range(0,1082):
    ys = y[:,i]
    ys = ys.reshape(260,1)
    sc = scaler.fit((ys))
    ys = sc.transform((ys))
    y[:,i] = np.transpose(ys)
    
    Xs = X[:,i]
    Xs = Xs.reshape(260,1)
    sc2 = scaler.fit((Xs))
    Xs = sc.transform((Xs))
    X[:,i] = np.transpose(Xs)
X = np.transpose(X)
y = np.transpose(y)


# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle  =True)

#Initializing Neural Network
model = Sequential()

# Adding the input layer and the first hidden layer
model.add(Dense(output_dim = 520, init = 'uniform', activation = 'relu', input_dim = 260))
#classifier.add(Dense(output_dim = 520, init = 'uniform', activation = 'relu'))
# Adding the second hidden layer
model.add(Dense(output_dim = 390, init = 'uniform', activation = 'relu'))
# Adding the output layer
model.add(Dense(output_dim = 260, init = 'uniform', activation = 'sigmoid'))

# Compiling Neural Network
model.compile(optimizer = Adam(lr = 0.0001), loss = 'mean_squared_error', metrics = ['accuracy'])

# Fitting our model 
history = model.fit(X_train, y_train, validation_data=(X_test,y_test), batch_size = 5, nb_epoch = 1000)
y_pred = model.predict(X_test)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")




# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.figure("X_test")
plt.plot(np.transpose(X_test[6]))
plt.figure("Y_test")
plt.plot(np.transpose(y_test[6]))
plt.figure("y_pred")
plt.plot(np.transpose(y_pred[6]))
plt.show()

data = pd.read_csv("./test.csv", engine='python')
series = data.values
print data

# Specify the data 
X=series[:,0:260]

# 0 to 1
y=series[:,260:520]
y = np.transpose(y)
X = np.transpose(X)
ys = y[:,0]
ys = ys.reshape(260,1)
sc = scaler.fit((ys))
ys = sc.transform((ys))
y[:,0] = np.transpose(ys)

Xs = X[:,0]
Xs = Xs.reshape(260,1)
sc2 = scaler.fit((Xs))
Xs = sc.transform((Xs))
X[:,0] = np.transpose(Xs)
X = np.transpose(X)
y = np.transpose(y)

y_pred = model.predict(X)
plt.figure("X")
plt.plot(np.transpose(X))
plt.figure("Y")
plt.plot(np.transpose(y))
plt.figure("y_pred")
plt.plot(np.transpose(y_pred))
plt.show()

# CrossValidation
"""
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
from keras.wrappers.scikit_learn import KerasClassifier
def create_network():
    
    # Start neural network
    network = Sequential()

    # Add fully connected layer with a ReLU activation function
    network.add(Dense(units=520, activation='relu', input_shape=(260,)))

    # Add fully connected layer with a ReLU activation function
    network.add(Dense(units=390, activation='relu'))

    # Add fully connected layer with a sigmoid activation function
    network.add(Dense(units=260, activation='sigmoid'))

    # Compile neural network
    network.compile(loss='mean_squared_error', # Cross-entropy
                    optimizer='Adam', # Root Mean Square Propagation
                    metrics=['accuracy']) # Accuracy performance metric
    
    # Return compiled network
    return network

# Perform 5-fold cross validation
neural_network = KerasClassifier(build_fn=create_network, 
                                 epochs=100, 
                                 batch_size=50)
scores = cross_val_score(neural_network, X, y, cv=5)
print scores
"""


