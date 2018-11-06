from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
import numpy as np
import pandas as pd

tenniscsv = pd.read_csv("tennis.csv")
# print(tenniscsv)
# print(list(tenniscsv.columns))

mdata = tenniscsv.as_matrix()
lst = []
for x in mdata[:,0]:
	if x == 'rainy': lst.append(0)
	elif x == 'overcast': lst.append(1)
	elif x == 'sunny': lst.append(2)
mdata[:,0] = lst
lst = []
for x in mdata[:,1]:
	if x == 'cool': lst.append(0)
	elif x == 'mild': lst.append(1)
	elif x == 'hot': lst.append(2)
mdata[:,1] = lst
mdata[:,2] = LabelEncoder().fit_transform(mdata[:,2])
mdata[:,3] = LabelEncoder().fit_transform(mdata[:,3])
mdata[:,4] = LabelEncoder().fit_transform(mdata[:,4])

tennis = pd.DataFrame(data=np.int_(mdata[:,:]), columns=list(tenniscsv.columns), index=[i for i in range(0,len(mdata))])
target = tennis.loc[:,'play']
data = tennis.loc[:,:'windy']

# print(data)
# print(target)

# model = Sequential()
# model.add(Dense(12, input_dim=8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.fit(X, Y, epochs=150, batch_size=10)

# scores = model.evaluate(X, Y)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))