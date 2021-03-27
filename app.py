#Basic imports
import numpy as np
import pandas as pd

#import machine learning algorithm
from sklearn.ensemble import GradientBoostingRegressor

#import train test split for splitting the data
from sklearn.model_selection import train_test_split
import warnings


import pickle
warnings.filterwarnings("ignore")

#import data
df = pd.read_csv("file1.csv")
df = np.array(df) #convert dataframe into array


x = df[:,1:-1] #selection of features
y = df[:,-1] #selection of target
y = y.astype('float')
x = x.astype('float')

#splitting the data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=20)

#model initialization
model = GradientBoostingRegressor()

#model trainning
model.fit(X_train, y_train)



#b = log_reg.predict_proba(final)

# model dump as pickle file
pickle.dump(model,open('model.pkl','wb'))
model1=pickle.load(open('model.pkl','rb'))

