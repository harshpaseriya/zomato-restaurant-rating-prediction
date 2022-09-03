
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import logging

data = pd.read_csv('G:\Complete Python Module\Projects\Restorent_rating\z_r_r\zomato.csv')

logging.basicConfig(filename='logs/model_development.txt',
                   filemode='a',format='%(asctime)s %(message)s',
                   datefmt="%Y-%m-%d %H:%M:%S")

logging.warning("Data Loading")

# In[3]:


sorted(data)


# In[4]:


data.head()


# In[5]:


data.isnull().sum()


# In[6]:


data.info()


# In[7]:


data=data.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type','listed_in(city)':'city'})


# In[8]:


sorted(data)
data.rest_type.unique()


# In[9]:


data.dropna(inplace=True)
data.isnull().sum()


# In[10]:


X = data.copy()

le = LabelEncoder()  
X.online_order = le.fit_transform(X.online_order)
X.book_table = le.fit_transform(X.book_table)
X['online_order'].unique()
X['book_table'].unique()
X.online_order.astype(float)
X.book_table.astype(float)
X.book_table


# In[11]:


X.rate.head()
X.rate.unique()


# In[12]:


X = X.loc[X.rate !='NEW']

X = X.loc[X.rate !='-'].reset_index(drop=True)

X.rate= X.rate.astype(str)

X.rate=X.rate.apply(lambda x : x.replace('/5',''))
X.rate=X.rate.astype(float)
X.rate


# In[13]:


X.cost = X.cost.astype(str)
X.cost = X.cost.apply(lambda x : x.replace(',',''))
X.cost = X.cost.astype(float)
X.cost


# In[14]:


X.votes.astype(float)


# In[15]:


X.drop_duplicates(keep='first',inplace = True)


# In[16]:


X.head()

logging.basicConfig(filename='logs/model_development.txt',
                   filemode='a',format='%(asctime)s %(message)s',
                   datefmt="%Y-%m-%d %H:%M:%S")

logging.warning("Feature Engineering done")

# <h2>Data visualization <h2>

# In[17]:


sns.countplot(data['city'])
sns.countplot(data['city']).set_xticklabels(sns.countplot(data['city']).get_xticklabels(), rotation=90, ha="right")
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('Location')


# <h4><i> Location-Rating <i><h4>

# In[18]:


loc_plt=pd.crosstab(X['rate'],X['city'])
loc_plt.plot(kind='bar',stacked=True);
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('Location - Rating',fontsize=15,fontweight='bold')
plt.ylabel('Location',fontsize=10,fontweight='bold')
plt.xlabel('Rating',fontsize=10,fontweight='bold')
plt.xticks(fontsize=10,fontweight='bold')
plt.yticks(fontsize=10,fontweight='bold');
plt.legend().remove();


# <h4><i> Type-Rating<i><h4>
# 

# In[19]:


type_plt=pd.crosstab(X['rate'],data['type'])
type_plt.plot(kind='bar',stacked=True);
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('Type - Rating',fontsize=15,fontweight='bold')
plt.ylabel('Type',fontsize=10,fontweight='bold')
plt.xlabel('Rating',fontsize=10,fontweight='bold')
plt.xticks(fontsize=10,fontweight='bold')
plt.yticks(fontsize=10,fontweight='bold');


# In[20]:


sns.boxplot(x=X.online_order, y =X.rate ,data = X)


# In[21]:


sns.boxplot(x=X.book_table, y =X.rate ,data = X)


# <h4><i>The above box plot also helps us look into the outliers. This box plot is
# regarding how table booking availability is seen in restaurants with
# rating over 4.
# <i><h4>

# <h4><i> Restuarants delivering online or not <i><h4>

# In[22]:


sns.countplot(x='online_order',data=data)
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('Restaurants delivering online or not')


# <h4><i> Restuarants allowing table booking or not<i><h4>

# In[23]:


sns.countplot(data['book_table'])
fig = plt.gcf()
fig.set_size_inches(6,6)
plt.title('Restaurants allowing table booking or not')


# <h4><i> Type of Service <i><h4>

# In[24]:


sns.countplot(X['type'])
sns.countplot(X['type']).set_xticklabels(sns.countplot(X['type']).get_xticklabels(), rotation=90, ha="right")
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('Type of Service')


# In[25]:


temp =X.name.value_counts()
temp = temp.head(10).plot(kind = 'barh',color='cyan')
temp.set_title('Top 10 restuarants in Bangalore')


# 
# <h4> <i>
#     Cost and Rate distribution according to online ordering and table booking <i> <h4>

# In[26]:


sns.scatterplot(data = X ,x='cost', y='rate',hue='online_order',
    sizes=(20, 200), legend="full")


# In[27]:


sns.scatterplot(data = X ,x='cost', y='rate',hue='book_table',
    sizes=(20, 200), legend="full")


# <h4><i>The above scatterplots shows the correspondence between the
# cost, online ordering, bookings and rating of the restaurant.<h4><i>

# In[28]:


sns.countplot(X['cost'])
sns.countplot(X['cost']).set_xticklabels(sns.countplot(X['cost']).get_xticklabels(), rotation=90, ha="right")
fig = plt.gcf()
fig.set_size_inches(12,12)
plt.title('Cost of Restuarant')


# <h4><i>Restuarants with around 300 to 800 rupees avergae bill(two people) are high in number<i><h4>
logging.basicConfig(filename='logs/model_development.txt',
                    filemode='a',format='%(asctime)s %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")

logging.warning("Data Visulization is Done")


# In[29]:

logging.basicConfig(filename='logs/model_development.txt',
                    filemode='a',format='%(asctime)s %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")

logging.warning("Feature Selection is Done")



X = X.drop(['url','address','phone','location','city','menu_item'],axis=1)


# <h2> Key Findings <h2>

# In[30]:


X.groupby('online_order').mean()


# In[31]:


X.groupby('book_table').mean()


# In[32]:


def Encode(X):
    for column in X.columns[~X.columns.isin(['rate', 'cost', 'votes'])]:
        X[column] = pd.factorize(X[column])[0]
    return X

enc = Encode(X.copy())
enc


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
y = X.rate
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

from sklearn.ensemble import  ExtraTreesRegressor
ET_Model=ExtraTreesRegressor(n_estimators = 120 , random_state=430)
ET_Model.fit(x_train,y_train)
y_predict=ET_Model.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_predict)


# In[44]:


sns.distplot(y_test-y_predict1)


# <h3>Random Forest Regressor<h3>

# In[52]:




# In[51]:





   
# 1. Linear Regressor shows approximately 23%
#   
# 2. Decision Tree Regressor shows approximately 83%

# 3. Random Forest Regressor shows approximately 87%
# 4. Extra Tree Regressor show approximately 94%

# In this model, we have considered various restaurants records with features like the name, average cost, locality, whether it accepts online order, can we book a table, type of restaurant.
# This model will help business owners predict their rating on the parameters considered in our model and improve the customer experience.<i><h4>
#       <h4>  <i>
# Different algorithms were used but in the end the final model is selected on Random Forest Regressor which gives the highest accuracy compared to others.<i><h4>

# In[47]:



