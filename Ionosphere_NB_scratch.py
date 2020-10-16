import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics as metrics

def load_data(path, header):
    df = pd.read_csv(path, header=header)
    return df

class Naive_Bayes():
	def __init__(self):
		self.datadict={}
               
	def fit(self,x_train,y_train):
				
		self.x_train=x_train
		self.y_train=y_train
		self.datadict[0]=np.array([[]])
		self.datadict[1]=np.array([[]])
		self.datadict=self.gendata(self.datadict,self.x_train,self.y_train)
		self.datadict[0]=np.transpose(self.datadict[0])
		self.datadict[1]=np.transpose(self.datadict[1])
		self.mean_0=np.mean(self.datadict[0],axis=0)
		self.mean_1=np.mean(self.datadict[1],axis=0)
		self.std_0=np.std(self.datadict[0],axis=0)
		self.std_1=np.std(self.datadict[1],axis=0)

	def gendata(self,datadict,x_train,y_train):
		set_one=True
		set_zero=True
		for i in range(y_train.shape[0]):
			x_temp=x_train[i,:].reshape(x_train[i,:].shape[0],1)
			if y_train[i]==1:
				if set_one==True:
					datadict[1]=x_temp
					set_one=False
				else:
					datadict[1]=np.append(datadict[1],x_temp,axis=1)
			elif y_train[i]==0:
				if set_zero==True:
					datadict[0]=x_temp
					set_zero=False
				else:
					datadict[0]=np.append(datadict[0],x_temp,axis=1)
		return datadict   
        
	def predict(self,x_test):
        
		p1=self.postprob(x_test,self.datadict[1],self.mean_1,self.std_1)
		p0=self.postprob(x_test,self.datadict[0],self.mean_0,self.std_0)
		return (p1>p0)

	def postprob(self,x,x_trainclass,mean_,std_):
			            
		p=np.prod(self.likelihood(x,mean_,std_),axis=1)
		p=p*(x_trainclass.shape[0]/self.x_train.shape[0])
		return p

	def likelihood(self,x,mean,sigma):
		for i in range(len(sigma)):
			if sigma[i] == 0:
				sigma[i] = 1
		return np.exp(-(x-mean)**2/(2*sigma**2))*(1/(np.sqrt(2*np.pi)*sigma))

	def calc_accuracy(self, ytest, pred):
		

		return np.mean(ytest == pred)

	def crossvalidation(self, xtrain, ytain, k, alpha = 0.01, iter = 50000, eps = 0.01):

		size = int(len(xtrain)/k)
		cv_accuracy = 0
		z=0

		for i in range(k):

			valstart = i*size
			valend = valstart + size

			if i!=(k-1):
				valend = size

				xval = xtrain[:valend,:]
				yval = ytrain[:valend]

				kxtrain = xtrain[valend:,:]
				kytrain = ytrain[valend:]

			else:
		
				xval = xtrain[valstart:,:]
				yval = ytrain[valstart:]

				kxtrain = xtrain[:valstart,:]
				kytrain = ytrain[:valstart]

				kxtrain = np.concatenate((xtrain[:valstart,:],xtrain[valend:,:]), axis = 0)
				kytrain = np.concatenate((ytrain[:valstart],ytrain[valend:]))

			w_kfold = self.fit(kxtrain, kytrain)
			#print(w_kfold)
			predy = self.predict(xval)
			cv_accuracy = cv_accuracy + self.calc_accuracy(yval,predy)
			
			#print(cv_accuracy)
			
		cv_accuracy = cv_accuracy / k

		return cv_accuracy


if __name__ == "__main__":
    # load the data from the file
	data = load_data("ionosphere.csv", None)
	

	for i in range(0,351):
		if data.loc[i,34] == 'g':
			data.loc[i,34] = 1
		else:
			data.loc[i,34] = 0

	data = data.drop(columns = [1])
	#print(np.std(data.iloc[:,1]))
	train_data = data.sample(frac = 0.8)

	xtrain = np.array(train_data.iloc[:,:-1])
	ytrain = np.array(train_data.iloc[:,-1])	
	test_data = data.drop(train_data.index)
	xtest = np.array((test_data.iloc[:,:-1]))
	ytest = np.array((test_data.iloc[:,-1]))


	nb=Naive_Bayes()

	nb.fit(xtrain,ytrain)

	pred=nb.predict(xtest)
	pred.astype(int)
	

	accuracy = nb.calc_accuracy(pred,ytest)
	print("Accuracy is:", accuracy)

	#Function call for k-fold validation

	#accuracy_kfold = nb.crossvalidation(xtrain, ytrain, 5)
	#print("Accuracy using k-fold is:", accuracy_kfold)
'''
	#No of instances and accuracy plot
	instance_vector = [10, 50, 100, 200, 300]
	accuracy = []

	for k in instance_vector:
		xtr = np.array(train_data.iloc[:k,:-1])
		ytr = np.array(train_data.iloc[:k,-1])
		xte = np.array((test_data.iloc[:k,:-1]))
		yte = np.array((test_data.iloc[:k,-1]))
		#w = np.array(np.transpose(np.zeros((xtrain.shape[1]))[np.newaxis]))
		#LR = Logistic_Regression(w)
		nb.fit(xtr, ytr)
		p = nb.predict(xte)
		a1 = nb.calc_accuracy(yte, p)
		accuracy.append(a1)
		print(accuracy)

	print(accuracy)
'''