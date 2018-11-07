from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
import numpy as np
import pandas as pd

tenniscsvNom = pd.read_csv("tennisNominal.csv")
tenniscsvCon = pd.read_csv("tennisContinue.csv")

# Nominal weather data preprocessing
label = tenniscsvNom.loc[:,'play']
tenniscsvNom = tenniscsvNom.drop('play',axis=1)
tenniscsvNom = pd.get_dummies(tenniscsvNom, columns=['outlook'], prefix=['outlook'])
tenniscsvNom = tenniscsvNom.join(label)

mdata = tenniscsvNom.as_matrix()
lst = []
for x in mdata[:,0]:
	if x == 'cool': lst.append(0)
	elif x == 'mild': lst.append(1)
	elif x == 'hot': lst.append(2)
mdata[:,0] = lst
mdata[:,1] = LabelEncoder().fit_transform(mdata[:,1])
mdata[:,2] = LabelEncoder().fit_transform(mdata[:,2])
mdata[:,6] = LabelEncoder().fit_transform(mdata[:,6])

tennis = pd.DataFrame(data=np.int_(mdata[:,:]), columns=list(tenniscsvNom.columns), index=[i for i in range(0,len(mdata))])
targetNom = tennis.loc[:,'play']
dataNom = tennis.loc[:,:'outlook_sunny']
print("Nominal Weather")
print(dataNom)
print(targetNom)

# Continue weather data preprocessing
label = tenniscsvCon.loc[:,'play']
tenniscsvCon = tenniscsvCon.drop('play',axis=1)
tenniscsvCon = pd.get_dummies(tenniscsvCon, columns=['outlook'], prefix=['outlook'])
tenniscsvCon = tenniscsvCon.join(label)

mdata = tenniscsvCon.as_matrix()
mdata[:,2] = LabelEncoder().fit_transform(mdata[:,2])
mdata[:,6] = LabelEncoder().fit_transform(mdata[:,6])

tennis = pd.DataFrame(data=np.int_(mdata[:,:]), columns=list(tenniscsvCon.columns), index=[i for i in range(0,len(mdata))])
targetCon = tennis.loc[:,'play']
dataCon = tennis.loc[:,:'outlook_sunny']
print("Continue Weather")
print(dataCon)
print(targetCon)

# Classify nominal weather
model = Sequential()
model.add(Dense(12, input_dim=6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(dataNom, targetNom, test_size=0.1)
model.fit(X_train, y_train, epochs=150, batch_size=5)
nominalScores = model.evaluate(X_test, y_test)

# Classify continue weather
model = Sequential()
model.add(Dense(12, input_dim=6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(dataCon, targetCon, test_size=0.1)
model.fit(X_train, y_train, epochs=150, batch_size=5)
continueScores = model.evaluate(X_test, y_test)

# Print scores
print("\nnominal weather acc: %.2f%%" % (nominalScores[1]*100))
print("\ncontinue weather acc: %.2f%%" % (continueScores[1]*100))