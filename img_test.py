#!/usr/bin/env python

import numpy as np
import sys
import pandas as pd
from skimage import io
from skimage import transform as trans
from sklearn.neural_network import MLPClassifier as NN
from sklearn.model_selection import train_test_split

labels = ['Noise', 'Has Hot Dogs', 'Ketchup', 'Mustard', 'Onion', 'Relish', 'Onion', 'Chili', 'Cheese']

#Get the data
print ("Reading CSV...")
data = pd.read_csv(filepath_or_buffer="hot_dog_data.csv", nrows=30)
X = data.values[1:,0]
Y = data.values[1:,1:8]

#convert the images to RGB number arrays
print ('Converting Images...')
img_converts = []
for line in X:
  img = io.imread("./images/"+line)
  img = trans.resize(img,(300,400), mode='constant')
  img_converts.append(img)
X = np.array(img_converts)

# Split into train and test vars
trainX, testX, trainY, testY = train_test_split(X,Y, test_size=0.17)

# Reshape the image arrays into 2-D arrays so it will fit the model
xa, xb, xc, xd = trainX.shape
d2_trainX = np.asarray(trainX.reshape((xa, xb*xc*xd)), dtype=np.float)

xe, xf, xg, xh = testX.shape
d2_testX = np.asarray(testX.reshape((xe, xf*xg*xh)), dtype=np.float)

clf = NN(solver='lbfgs',hidden_layer_sizes=(5,5,5), random_state=42, verbose=True)

# Recast the Y data so the fit won't get a label mismatch
trainY = np.asarray(trainY, dtype=np.integer)
testY = np.asarray(testY, dtype=np.integer)

print ('The machine is learning...')
clf.fit(d2_trainX, trainY)

print ('Predicting...')
count = 1,

for line in clf.predict(d2_testX):
  for i in range(len(line)):
    if int(line[i]) == 1:
      print labels[i] + ',' ,
  print("")

print 'Calculating Accuracy...'
count = 1
scores = clf.score(d2_testX, testY)
print (scores)

sys.exit()
