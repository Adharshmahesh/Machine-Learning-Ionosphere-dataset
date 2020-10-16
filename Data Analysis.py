import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

i=0
labels = ['good','bad']

def load_data(path, header):
    df = pd.read_csv(path, header=header)
    return df


if __name__ == "__main__":
    # load the data from the file
    data = load_data("ionosphere.csv", None)

#print(np.mean(data[1])) #gives mean of any column
#print(np.median(data[34])) #gives median of any column
#print(stats.mode(data[34])) #gives mode of any column
#print(data.loc[1,:]) #print values of any element in the data frame
#print(data.loc[:,1].isnull()) #to check if the data in that location is null
#print(data.isnull()) #to check if the whole data has null or not



z=data.describe()
print(z)
plt.hist(data.loc[:,34], bins = 20)
#plt.xlabel('')
plt.ylabel('Frequency')
plt.title('No of good and bad pulses')
plt.figure()
corr = data.corr()
sns.heatmap(corr, annot=True, linewidths=1.0)

plt.show()

columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.695:
            print(i,j)
            
        else:
            #print(i)
            continue
#Correlation with output variable
cor_target = abs(corr.iloc[:,-1])#Selecting highly correlated features
relevant_features = cor_target[cor_target<0.05]
print(relevant_features)

#Pairwise scatter

fig, a = plt.subplots(2,2)
a[0][0].scatter(data.iloc[:,10], data.iloc[:,16])
a[0][1].scatter(data.iloc[:,12], data.iloc[:,14])
a[1][0].scatter(data.iloc[:,14], data.iloc[:,20])
a[1][1].scatter(data.iloc[:,14],data.iloc[:,16])

plt.show()
#x=data.loc[:,2]
#y=data.loc[:,3]
#plt.scatter(x,y)
#plt.show()
sns.distplot(data.iloc[:,31])
plt.show()


sns.boxplot(data=data.iloc[:,31])
plt.show()
