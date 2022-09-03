
import pre_processing as pro
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def Encode(X):
    for column in X.columns[~X.columns.isin(['rate', 'cost', 'votes'])]:
        X[column] = pd.factorize(X[column])[0]
    return X

enc = Encode(pro.X.copy())
enc

logging.basicConfig(filename='logs/model_development.txt',
                    filemode='a',format='%(asctime)s %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")

logging.warning("Standarddizing the data")


# <h3>Standardizing the data<h3>

# In[33]:


scaler = StandardScaler()


# In[34]:


x_fit=scaler.fit_transform(enc)


# In[35]:


enc=pd.DataFrame(x_fit,columns=enc.columns)
enc.head()


# In[36]:


enc.info()


# <h3>Correlation between different variables <h3>

# In[37]:


corr = enc.corr(method='kendall')
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True)
enc.columns


# <h3>Splitting the dataset<h3>

# In[38]:


x = enc.iloc[:,[1,2,4,5,6,7,8,9,10]]
y = pro.X.rate
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=42)
x_train.head()


# In[39]:


from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(x,y)


# In[40]:


feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(8).plot(kind='barh')
plt.show()


# <h3>Linear Regressor<h3>

# In[41]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[42]:


sns.distplot(y_test-y_pred)


# <h3>Decision Tree Regressor<h3>

# In[50]:


from sklearn.tree import DecisionTreeRegressor
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=0)
DTree=DecisionTreeRegressor(min_samples_leaf=.0001)
DTree.fit(x_train,y_train)
y_predict1=DTree.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_predict1)


# In[44]:


sns.distplot(y_test-y_predict1)
from sklearn.ensemble import RandomForestRegressor
RForest=RandomForestRegressor(n_estimators=500,random_state=430,min_samples_leaf=.0001)
RForest.fit(pro.x_train,pro.y_train)
y_predict=RForest.predict(pro.x_test)
from sklearn.metrics import r2_score
r2_score(pro.y_test,y_predict)

sns.distplot(pro.y_test-y_predict)

from sklearn.ensemble import  ExtraTreesRegressor
ET_Model=ExtraTreesRegressor(n_estimators = 120 , random_state=430)
ET_Model.fit(x_train,y_train)
y_predict=ET_Model.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_predict)

sns.distplot(y_test-y_predict)

import pickle


# In[53]:


file = open('ET_model.pkl','wb')
pickle.dump(ET_Model,file)

print("Done")

logging.basicConfig(filename='logs/model_development.txt',
                    filemode='a',format='%(asctime)s %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")

logging.warning("ExtraTree Regressor done !!!")

ET_Model.predict(x_test)
# In[165]: