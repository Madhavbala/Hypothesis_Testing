import pandas as pd
import numpy as np
from numpy import sqrt,abs,round
from scipy.stats import norm

# Null Hypothesis : Temperature does not affect the COVID Outbreak
# Alternative Hypothesis : Temperature does effect the COVID Outbreak

dataset = pd.read_csv('C:\\Users\\madhavan.bala\\Downloads\\dataset.csv')
print(dataset.columns)

dataset['Temp_cat'] = dataset['Temprature'].apply(lambda x : 0 if x < 24 else 1)
dataset_modified = dataset[['Confirmed','Temp_cat']]
print(dataset_modified)

d1 = dataset_modified[(dataset_modified['Temp_cat']==1)]['Confirmed']
d2 = dataset_modified[(dataset_modified['Temp_cat']==0)]['Confirmed']

mean_1 , mean_2 = d1.mean() , d2.mean()
sd1 , sd2 = d1.std() , d2.std()
n1 , n2 = d1.shape , d2.shape

print("Mean: ",mean_1 , mean_2)
print("Standard_deviation : " , sd1 , sd2)
print("Shape : ", n1,n2)


def model(X1,X2,sigma1,sigma2,N1,N2):
    over_sigma = sqrt(sigma1**2/N1 + sigma2**2/N2)
    z = (X1 - X2)/over_sigma
    p_val = 2*(1-norm.cdf(abs(z)))
    return z, p_val
z,p = model(mean_1,mean_2,sd1,sd2,n1,n2)

z_score = np.round(z,8)
p_value = np.round(p,6)
