import numpy as np
import pandas as pd
import scipy
import sklearn
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

def load_data(path, header):
    df = pd.read_csv(path, header=header)
    return df


if __name__ == "__main__":
    # load the data from the file
    data = load_data("ionosphere.csv", None)


for i in range(0,351):
		if data.loc[i,34] == 'g':
			data.loc[i,34] = 1
		else:
			data.loc[i,34] = 0

X=data.iloc[:,:32]
y=data.iloc[:,33]
y=y.astype('int')



from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
print(metrics.accuracy_score(y_test, y_pred))

#Lasso penalty
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(x_train,y_train)
train_score=lasso.score(x_train,y_train)
test_score=lasso.score(x_test,y_test)
coeff_used = np.sum(lasso.coef_==0)
#parameters = lasso.get_params(deep = True)

print("coeff_used is", coeff_used)
#print(parameters)

