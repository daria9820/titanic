


import pandas as pd
import numpy as np





data = pd.read_csv('titanic.csv')





data




columns_target = ['Survived'] # наша целевая колонка

columns_train = ['Pclass', 'Sex', 'Age', 'Fare']




X = data[columns_train]
Y = data[columns_target]





X['Sex'].isnull().sum()


# In[186]:


X['Pclass'].isnull().sum()





X['Fare'].isnull().sum()





X['Age'].isnull().sum()





X['Age'] = X['Age'].fillna(X['Age'].mean())





X['Age'].isnull().sum()





d={'male':0, 'female':1} 




X['Sex'] = X['Sex'].apply(lambda x:d[x])





X['Sex'].head() 





from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)





from sklearn import svm




predmodel = svm.LinearSVC()





predmodel.fit(X_train, Y_train)





predmodel.predict(X_test[0:10])





predmodel.score(X_test,Y_test)







