import numpy as np
import pandas as pd
import scipy
#from loaddata import data
import sklearn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import datasets

def load_data(path, header):
    df = pd.read_csv(path, header=header)
    return df


if __name__ == "__main__":
    # load the data from the file
    data = load_data("Ionosphere_dataset.csv", None)


for i in range(0,351):
		if data.loc[i,34] == 'g':
			data.loc[i,34] = 1
		else:
			data.loc[i,34] = 0


X=data.iloc[:,:32]
y=data.iloc[:,33]
y=y.astype('int')
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) 

from sklearn.linear_model import LogisticRegression   
# create logistic regression object 
reg = LogisticRegression() 
   
# train the model using the training sets 
reg.fit(X_train, y_train) 
  
# making predictions on the testing set 
y_pred = reg.predict(X_test) 

w = reg.coef_ 
# comparing actual response values (y_test) with predicted response values (y_pred) 
#print(w)
print("Logistic Regression model accuracy(in %):",  
metrics.accuracy_score(y_test, y_pred)*100) 
