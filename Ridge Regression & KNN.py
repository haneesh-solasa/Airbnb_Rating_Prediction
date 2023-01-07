#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from googletrans import Translator
import langdetect
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


# In[2]:


get_ipython().run_line_magic('store', '-r final_df')
get_ipython().run_line_magic('store', '-r y_1')
get_ipython().run_line_magic('store', '-r y_2')
get_ipython().run_line_magic('store', '-r y_3')
get_ipython().run_line_magic('store', '-r y_4')
get_ipython().run_line_magic('store', '-r y_5')
get_ipython().run_line_magic('store', '-r y_6')
get_ipython().run_line_magic('store', '-r y_7')


# In[3]:


final_df = final_df
y_1 = y_1
y_2 = y_2
y_3 = y_3
y_4 = y_4
y_5 = y_5
y_6 = y_6
y_7 = y_7


# In[4]:


get_ipython().run_line_magic('store', '-d final_df')
get_ipython().run_line_magic('store', '-d y_1')
get_ipython().run_line_magic('store', '-d y_2')
get_ipython().run_line_magic('store', '-d y_3')
get_ipython().run_line_magic('store', '-d y_4')
get_ipython().run_line_magic('store', '-d y_5')
get_ipython().run_line_magic('store', '-d y_6')
get_ipython().run_line_magic('store', '-d y_7')


# ## Fixing the Max Features as 10000 as it has the least MSE

# In[45]:


X= final_df

vectorizer1 = TfidfVectorizer(max_features=10000, ngram_range=(1,3))    
tfidf_1 = vectorizer1.fit_transform(X['description'])
df_tfidf = pd.DataFrame(tfidf_1.toarray(), columns=['description_' + x for x in vectorizer1.get_feature_names()])
X = pd.concat([X, df_tfidf], axis=1)
X.drop(columns=['description'], axis=1, inplace=True)
    
vectorizer2 = TfidfVectorizer(max_features=10000, ngram_range=(1,3)) 
tfidf_2 = vectorizer2.fit_transform(X['neighborhood_overview'])
df_tfidf = pd.DataFrame(tfidf_2.toarray(), columns=['neihborhood_overview_' + x for x in vectorizer2.get_feature_names()])
X = pd.concat([X, df_tfidf], axis=1)
X.drop(columns=['neighborhood_overview'], axis=1, inplace=True)
    
vectorizer3 = TfidfVectorizer(max_features=10000, ngram_range=(1,3)) 
tfidf_3 = vectorizer3.fit_transform(X['translated'])
df_tfidf = pd.DataFrame(tfidf_3.toarray(), columns=['translated_' + x for x in vectorizer3.get_feature_names()])
X = pd.concat([X, df_tfidf], axis=1)
X.drop(columns=['translated'], axis=1, inplace=True)


# In[46]:


y = y_1

model_1 = Ridge()
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}


grid_search_ridge = GridSearchCV(model_1, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error',verbose = 2,return_train_score=True)

grid_search_ridge.fit(X_train, Y_train)

print(f"Best regularization: {grid_search_ridge.best_estimator_.alpha}")
results_df3 = pd.DataFrame.from_dict(grid_search_ridge.cv_results_)
results_df3.to_csv('GridSearchCV_Ridge_10000.csv', index=False)


# In[141]:


plt.figure(figsize=(6,4), dpi=200)
results_df3['param_alpha'] = results_df3['param_alpha'].astype('int')
acc = -1*results_df3["mean_train_score"]
val_acc = -1*results_df3["mean_test_score"]
mean = -1*results_df3['mean_test_score']
std = results_df3['std_test_score']


epochs = range(len(acc))
plt.plot(results_df3["param_alpha"], acc, label='Training set', color='mediumseagreen', linestyle='-', lw=2)
#plt.plot(results_df3["param_alpha"], val_acc, label='Validation set', color='orangered', linestyle='--', lw=2)
plt.errorbar(results_df3["param_alpha"],mean,yerr=std,ecolor='red',fmt='-o',label = 'Red-Std_Error and Blue=MSE')
plt.title('5-fold GridSearchCV: Alpha vs MSE for \n Ridge Regression', fontsize=18, pad=20)
plt.legend([])



plt.xlabel('Alpha', fontsize=16, labelpad=20)
plt.ylabel('Mean Squared Error', fontsize=16, labelpad=20)
plt.xlim([-1,200])
plt.tick_params(labelsize=14)



plt.figlegend(loc='upper right', ncol=1, labelspacing=0.3,
              title_fontsize=12, fontsize=10, bbox_to_anchor=(0.9, 0.9),
              handletextpad=0.6, frameon=True)
plt.show()


# In[48]:


mean = -1*results_df3['mean_test_score']
std = results_df3['std_test_score']
plt.figure(figsize=(6,4), dpi=200)
plt.errorbar(results_df3["param_alpha"],mean,yerr=std,ecolor='red',fmt='-o')
plt.legend(["Red-Std_Error and Blue=MSE"],loc=4) 
plt.xlim([-5,200])
plt.xlabel('Alpha', fontsize=16, labelpad=20)
plt.ylabel('Mean Squared Error', fontsize=16, labelpad=20)
plt.title('Ridge: 5 fold CV', fontsize=18, pad=20)
plt.show()


# ## Model 1

# In[49]:


y = y_1
ridge_model_1 = Ridge()
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X, y, test_size=0.2, random_state=42)
ridge_model_1.fit(X1_train,Y1_train)
Y1_pred = ridge_model_1.predict(X1_test)
score1 = mean_squared_error(Y1_pred,Y1_test)
mae1 = mean_absolute_error(Y1_pred,Y1_test)


# In[50]:


importance = ridge_model_1.coef_
feature_names = list(X1_train)
indices = np.where(np.abs(importance) > 0.1)[0]
feature_filtered = [feature_names[i] for i in indices]
df_filtered = X.loc[:, feature_filtered]

final_ridge_model_1 = Ridge(alpha = 1)
X1_train, X1_test, Y1_train, Y1_test = train_test_split(df_filtered, y, test_size=0.2, random_state=42)
final_ridge_model_1.fit(X1_train,Y1_train)
ypred = final_ridge_model_1.predict(X1_test)
score1_final = mean_squared_error(ypred,Y1_test)
mae1_final = mean_absolute_error(ypred,Y1_test)


# In[84]:


top_n = 20
sorted_indexes1 = np.abs(importance).argsort()[::-1][:top_n]

# Plot the feature importance
plt.figure(figsize=(6,4), dpi=200)
plt.bar(range(top_n), importance[sorted_indexes1])
plt.xticks(range(top_n), [feature_names[i] for i in sorted_indexes1], rotation=90)
plt.ylabel('Coefficient value', fontsize=16, labelpad=20)
plt.title('Top {} feature importance'.format(top_n), fontsize=18, pad=20)
plt.show()


# In[52]:


print(score1)
print(score1_final)
print(mae1)
print(mae1_final)


# In[81]:


mae1_final = mean_absolute_error(ypred,Y1_test)
mae1_final


# ## Model 2

# In[56]:


y = y_2
ridge_model_2 = Ridge()
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X, y, test_size=0.2, random_state=42)
ridge_model_2.fit(X2_train,Y2_train)
Y2_pred = ridge_model_2.predict(X2_test)
score2 = mean_squared_error(Y2_pred,Y2_test)
mae2 = mean_absolute_error(Y2_pred,Y2_test)


# In[57]:


importance2 = ridge_model_2.coef_
feature_names2 = list(X2_train)
indices2 = np.where(np.abs(importance2) > 0.1)[0]
feature_filtered2 = [feature_names2[i] for i in indices2]
df_filtered2 = X.loc[:, feature_filtered2]

final_ridge_model_2 = Ridge(alpha = 1)
X2_train, X2_test, Y2_train, Y2_test = train_test_split(df_filtered2, y, test_size=0.2, random_state=42)
final_ridge_model_2.fit(X2_train,Y2_train)
ypred2 = final_ridge_model_2.predict(X2_test)
score2_final = mean_squared_error(ypred2,Y2_test)
mae2_final = mean_absolute_error(ypred2,Y2_test)


# In[58]:


print(score2)
print(score2_final)
print(mae2)
print(mae2_final)


# In[82]:


top_n = 20
sorted_indexes2 = np.abs(importance2).argsort()[::-1][:top_n]

# Plot the feature importance
plt.figure(figsize=(6,4), dpi=200)
plt.bar(range(top_n), importance2[sorted_indexes2])
plt.xticks(range(top_n), [feature_names2[i] for i in sorted_indexes2], rotation=90)
plt.ylabel('Coefficient value', fontsize=16, labelpad=20)
plt.title('Top {} feature importance'.format(top_n), fontsize=18, pad=20)
plt.show()


# ## Model 3

# In[60]:


y = y_3
ridge_model_3 = Ridge()
X3_train, X3_test, Y3_train, Y3_test = train_test_split(X, y, test_size=0.2, random_state=42)
ridge_model_3.fit(X3_train,Y3_train)
Y3_pred = ridge_model_3.predict(X3_test)
score3 = mean_squared_error(Y3_pred,Y3_test)
mae3 = mean_absolute_error(Y3_pred,Y3_test)


# In[62]:


importance3 = ridge_model_3.coef_
feature_names3 = list(X3_train)
indices3 = np.where(np.abs(importance3) > 0.1)[0]
feature_filtered3 = [feature_names3[i] for i in indices3]
df_filtered3 = X.loc[:, feature_filtered3]

final_ridge_model_3 = Ridge(alpha = 1)
X3_train, X3_test, Y3_train, Y3_test = train_test_split(df_filtered3, y, test_size=0.3, random_state=42)
final_ridge_model_3.fit(X3_train,Y3_train)
ypred3 = final_ridge_model_3.predict(X3_test)
score3_final = mean_squared_error(ypred3,Y3_test)
mae3_final = mean_absolute_error(ypred3,Y3_test)


# In[63]:


print(score3)
print(score3_final)
print(mae3)
print(mae3_final)


# In[64]:


top_n = 20
sorted_indexes3 = np.abs(importance3).argsort()[::-1][:top_n]

# Plot the feature importance
plt.bar(range(top_n), importance3[sorted_indexes3])
plt.xticks(range(top_n), [feature_names3[i] for i in sorted_indexes3], rotation=90)
plt.ylabel('Coefficient value')
plt.title('Top {} feature importance'.format(top_n))
plt.show()


# ## Model 4

# In[65]:


y = y_4
ridge_model_4 = Ridge()
X4_train, X4_test, Y4_train, Y4_test = train_test_split(X, y, test_size=0.2, random_state=42)
ridge_model_4.fit(X4_train,Y4_train)
Y4_pred = ridge_model_4.predict(X4_test)
score4 = mean_squared_error(Y4_pred,Y4_test)
mae4 = mean_absolute_error(Y4_pred,Y4_test)


# In[66]:


importance4 = ridge_model_4.coef_
feature_names4 = list(X4_train)
indices4 = np.where(np.abs(importance4) > 0.1)[0]
feature_filtered4 = [feature_names4[i] for i in indices4]
df_filtered4 = X.loc[:, feature_filtered4]

final_ridge_model_4 = Ridge(alpha = 1)
X4_train, X4_test, Y4_train, Y4_test = train_test_split(df_filtered4, y, test_size=0.4, random_state=42)
final_ridge_model_4.fit(X4_train,Y4_train)
ypred4 = final_ridge_model_4.predict(X4_test)
score4_final = mean_squared_error(ypred4,Y4_test)
mae4_final = mean_absolute_error(ypred4,Y4_test)


# In[67]:


print(score4)
print(score4_final)
print(mae4)
print(mae4_final)


# In[68]:


top_n = 20
sorted_indexes4 = np.abs(importance4).argsort()[::-1][:top_n]

# Plot the feature importance
plt.bar(range(top_n), importance4[sorted_indexes4])
plt.xticks(range(top_n), [feature_names4[i] for i in sorted_indexes4], rotation=90)
plt.ylabel('Coefficient value')
plt.title('Top {} feature importance'.format(top_n))
plt.show()


# ## Model 5

# In[69]:


y = y_5
ridge_model_5 = Ridge()
X5_train, X5_test, Y5_train, Y5_test = train_test_split(X, y, test_size=0.2, random_state=42)
ridge_model_5.fit(X5_train,Y5_train)
Y5_pred = ridge_model_5.predict(X5_test)
score5 = mean_squared_error(Y5_pred,Y5_test)
mae5 = mean_absolute_error(Y5_pred,Y5_test)


# In[70]:


importance5 = ridge_model_5.coef_
feature_names5 = list(X5_train)
indices5 = np.where(np.abs(importance5) > 0.1)[0]
feature_filtered5 = [feature_names5[i] for i in indices5]
df_filtered5 = X.loc[:, feature_filtered5]

final_ridge_model_5 = Ridge(alpha = 1)
X5_train, X5_test, Y5_train, Y5_test = train_test_split(df_filtered5, y, test_size=0.5, random_state=42)
final_ridge_model_5.fit(X5_train,Y5_train)
ypred5 = final_ridge_model_5.predict(X5_test)
score5_final = mean_squared_error(ypred5,Y5_test)
mae5_final = mean_absolute_error(ypred5,Y5_test)


# In[71]:


print(score5)
print(score5_final)
print(mae5)
print(mae5_final)


# In[72]:


top_n = 20
sorted_indexes5 = np.abs(importance5).argsort()[::-1][:top_n]

# Plot the feature importance
plt.bar(range(top_n), importance5[sorted_indexes5])
plt.xticks(range(top_n), [feature_names5[i] for i in sorted_indexes5], rotation=90)
plt.ylabel('Coefficient value')
plt.title('Top {} feature importance'.format(top_n))
plt.show()


# ## Model 6

# In[73]:


y = y_6
ridge_model_6 = Ridge()
X6_train, X6_test, Y6_train, Y6_test = train_test_split(X, y, test_size=0.2, random_state=42)
ridge_model_6.fit(X6_train,Y6_train)
Y6_pred = ridge_model_6.predict(X6_test)
score6 = mean_squared_error(Y6_pred,Y6_test)
mae6 = mean_absolute_error(Y6_pred,Y6_test)


# In[74]:


importance6 = ridge_model_6.coef_
feature_names6 = list(X6_train)
indices6 = np.where(np.abs(importance6) > 0.1)[0]
feature_filtered6 = [feature_names6[i] for i in indices6]
df_filtered6 = X.loc[:, feature_filtered6]

final_ridge_model_6 = Ridge(alpha = 1)
X6_train, X6_test, Y6_train, Y6_test = train_test_split(df_filtered6, y, test_size=0.6, random_state=42)
final_ridge_model_6.fit(X6_train,Y6_train)
ypred6 = final_ridge_model_6.predict(X6_test)
score6_final = mean_squared_error(ypred6,Y6_test)
mae6_final = mean_absolute_error(ypred6,Y6_test)


# In[75]:


print(score6)
print(score6_final)
print(mae6)
print(mae6_final)


# In[76]:


top_n = 20
sorted_indexes6 = np.abs(importance6).argsort()[::-1][:top_n]

# Plot the feature importance
plt.bar(range(top_n), importance6[sorted_indexes6])
plt.xticks(range(top_n), [feature_names6[i] for i in sorted_indexes6], rotation=90)
plt.ylabel('Coefficient value')
plt.title('Top {} feature importance'.format(top_n))
plt.show()


# ## Model 7

# In[77]:


y = y_7
ridge_model_7 = Ridge()
X7_train, X7_test, Y7_train, Y7_test = train_test_split(X, y, test_size=0.2, random_state=42)
ridge_model_7.fit(X7_train,Y7_train)
Y7_pred = ridge_model_7.predict(X7_test)
score7 = mean_squared_error(Y7_pred,Y7_test)
mae7 = mean_absolute_error(Y7_pred,Y7_test)


# In[78]:


importance7 = ridge_model_7.coef_
feature_names7 = list(X7_train)
indices7 = np.where(np.abs(importance7) > 0.1)[0]
feature_filtered7 = [feature_names7[i] for i in indices7]
df_filtered7 = X.loc[:, feature_filtered7]

final_ridge_model_7 = Ridge(alpha = 1)
X7_train, X7_test, Y7_train, Y7_test = train_test_split(df_filtered7, y, test_size=0.7, random_state=42)
final_ridge_model_7.fit(X7_train,Y7_train)
ypred7 = final_ridge_model_7.predict(X7_test)
score7_final = mean_squared_error(ypred7,Y7_test)
mae7_final = mean_absolute_error(ypred7,Y7_test)


# In[79]:


print(score7)
print(score7_final)
print(mae7)
print(mae7_final)


# In[40]:


top_n = 20
sorted_indexes7 = np.abs(importance7).argsort()[::-1][:top_n]

# Plot the feature importance
plt.bar(range(top_n), importance7[sorted_indexes7])
plt.xticks(range(top_n), [feature_names7[i] for i in sorted_indexes7], rotation=90)
plt.ylabel('Coefficient value')
plt.title('Top {} feature importance'.format(top_n))
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# # K Nearest Neighbors Regression

# In[85]:


X= final_df

vectorizer1 = TfidfVectorizer(max_features=1500, ngram_range=(1,3))    
tfidf_1 = vectorizer1.fit_transform(X['description'])
df_tfidf = pd.DataFrame(tfidf_1.toarray(), columns=['description_' + x for x in vectorizer1.get_feature_names()])
X = pd.concat([X, df_tfidf], axis=1)
X.drop(columns=['description'], axis=1, inplace=True)
    
vectorizer2 = TfidfVectorizer(max_features=1500, ngram_range=(1,3)) 
tfidf_2 = vectorizer2.fit_transform(X['neighborhood_overview'])
df_tfidf = pd.DataFrame(tfidf_2.toarray(), columns=['neihborhood_overview_' + x for x in vectorizer2.get_feature_names()])
X = pd.concat([X, df_tfidf], axis=1)
X.drop(columns=['neighborhood_overview'], axis=1, inplace=True)
    
vectorizer3 = TfidfVectorizer(max_features=1500, ngram_range=(1,3)) 
tfidf_3 = vectorizer3.fit_transform(X['translated'])
df_tfidf = pd.DataFrame(tfidf_3.toarray(), columns=['translated_' + x for x in vectorizer3.get_feature_names()])
X = pd.concat([X, df_tfidf], axis=1)
X.drop(columns=['translated'], axis=1, inplace=True)


# ## Model 1

# In[88]:


y = y_1
KNN1 = KNeighborsRegressor(n_neighbors = 20)
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X, y, test_size=0.2, random_state=42)
KNN1.fit(X1_train,Y1_train)
Y1_pred = KNN1.predict(X1_test)
score1 = mean_squared_error(Y1_pred,Y1_test)
mae1 = mean_absolute_error(Y1_pred,Y1_test)


# In[89]:


filtered_df1 = X.loc[:, [col for col in feature_filtered if col in X.columns]]

final_KNN_1 = KNeighborsRegressor(n_neighbors = 20)
X1_train, X1_test, Y1_train, Y1_test = train_test_split(filtered_df1, y, test_size=0.2, random_state=42)
final_KNN_1.fit(X1_train,Y1_train)
ypred = final_KNN_1.predict(X1_test)
score1_final = mean_squared_error(ypred,Y1_test)
mae1_final = mean_absolute_error(ypred,Y1_test)


# In[90]:


print(score1)
print(mae1)
print(score1_final)
print(mae1_final)


# ## Model 2

# In[95]:


y = y_2
KNN2 = KNeighborsRegressor(n_neighbors = 20)
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X, y, test_size=0.2, random_state=42)
KNN2.fit(X2_train,Y2_train)
Y2_pred = KNN2.predict(X2_test)
score2 = mean_squared_error(Y2_pred,Y2_test)
mae2 = mean_absolute_error(Y2_pred,Y2_test)


# In[99]:


filtered_df2 = X.loc[:, [col for col in feature_filtered2 if col in X.columns]]

final_KNN_2 = KNeighborsRegressor(n_neighbors = 20)
X2_train, X2_test, Y2_train, Y2_test = train_test_split(filtered_df1, y, test_size=0.2, random_state=42)
final_KNN_2.fit(X2_train,Y2_train)
ypred2 = final_KNN_2.predict(X2_test)
score2_final = mean_squared_error(ypred2,Y2_test)
mae2_final = mean_absolute_error(ypred2,Y2_test)


# In[100]:


print(score2)
print(mae2)
print(score2_final)
print(mae2_final)


# ## Model 3

# In[107]:


y = y_3
KNN3 = KNeighborsRegressor(n_neighbors = 20)
X3_train, X3_test, Y3_train, Y3_test = train_test_split(X, y, test_size=0.3, random_state=42)
KNN3.fit(X3_train,Y3_train)
Y3_pred = KNN3.predict(X3_test)
score3 = mean_squared_error(Y3_pred,Y3_test)
mae3 = mean_absolute_error(Y3_pred,Y3_test)


# In[108]:


filtered_df3 = X.loc[:, [col for col in feature_filtered3 if col in X.columns]]

final_KNN_3 = KNeighborsRegressor(n_neighbors = 20)
X3_train, X3_test, Y3_train, Y3_test = train_test_split(filtered_df1, y, test_size=0.3, random_state=42)
final_KNN_3.fit(X3_train,Y3_train)
ypred3 = final_KNN_3.predict(X3_test)
score3_final = mean_squared_error(ypred3,Y3_test)
mae3_final = mean_absolute_error(ypred3,Y3_test)


# In[109]:


print(score3)
print(mae3)
print(score3_final)
print(mae3_final)


# ## Model 4

# In[111]:


y = y_4
KNN4 = KNeighborsRegressor(n_neighbors = 20)
X4_train, X4_test, Y4_train, Y4_test = train_test_split(X, y, test_size=0.4, random_state=42)
KNN4.fit(X4_train,Y4_train)
Y4_pred = KNN4.predict(X4_test)
score4 = mean_squared_error(Y4_pred,Y4_test)
mae4 = mean_absolute_error(Y4_pred,Y4_test)


# In[112]:


filtered_df4 = X.loc[:, [col for col in feature_filtered4 if col in X.columns]]

final_KNN_4 = KNeighborsRegressor(n_neighbors = 20)
X4_train, X4_test, Y4_train, Y4_test = train_test_split(filtered_df1, y, test_size=0.4, random_state=42)
final_KNN_4.fit(X4_train,Y4_train)
ypred4 = final_KNN_4.predict(X4_test)
score4_final = mean_squared_error(ypred4,Y4_test)
mae4_final = mean_absolute_error(ypred4,Y4_test)


# In[113]:


print(score4)
print(mae4)
print(score4_final)
print(mae4_final)


# ## Model 5

# In[114]:


y = y_5
KNN5 = KNeighborsRegressor(n_neighbors = 20)
X5_train, X5_test, Y5_train, Y5_test = train_test_split(X, y, test_size=0.5, random_state=42)
KNN5.fit(X5_train,Y5_train)
Y5_pred = KNN5.predict(X5_test)
score5 = mean_squared_error(Y5_pred,Y5_test)
mae5 = mean_absolute_error(Y5_pred,Y5_test)


# In[115]:


filtered_df5 = X.loc[:, [col for col in feature_filtered5 if col in X.columns]]

final_KNN_5 = KNeighborsRegressor(n_neighbors = 20)
X5_train, X5_test, Y5_train, Y5_test = train_test_split(filtered_df1, y, test_size=0.5, random_state=42)
final_KNN_5.fit(X5_train,Y5_train)
ypred5 = final_KNN_5.predict(X5_test)
score5_final = mean_squared_error(ypred5,Y5_test)
mae5_final = mean_absolute_error(ypred5,Y5_test)


# In[116]:


print(score5)
print(mae5)
print(score5_final)
print(mae5_final)


# ## Model 6

# In[117]:


y = y_6
KNN6 = KNeighborsRegressor(n_neighbors = 20)
X6_train, X6_test, Y6_train, Y6_test = train_test_split(X, y, test_size=0.6, random_state=42)
KNN6.fit(X6_train,Y6_train)
Y6_pred = KNN6.predict(X6_test)
score6 = mean_squared_error(Y6_pred,Y6_test)
mae6 = mean_absolute_error(Y6_pred,Y6_test)


# In[118]:


filtered_df6 = X.loc[:, [col for col in feature_filtered6 if col in X.columns]]

final_KNN_6 = KNeighborsRegressor(n_neighbors = 20)
X6_train, X6_test, Y6_train, Y6_test = train_test_split(filtered_df1, y, test_size=0.6, random_state=42)
final_KNN_6.fit(X6_train,Y6_train)
ypred6 = final_KNN_6.predict(X6_test)
score6_final = mean_squared_error(ypred6,Y6_test)
mae6_final = mean_absolute_error(ypred6,Y6_test)


# In[119]:


print(score6)
print(mae6)
print(score6_final)
print(mae6_final)


# ## Model 7

# In[120]:


y = y_7
KNN7 = KNeighborsRegressor(n_neighbors = 20)
X7_train, X7_test, Y7_train, Y7_test = train_test_split(X, y, test_size=0.7, random_state=42)
KNN7.fit(X7_train,Y7_train)
Y7_pred = KNN7.predict(X7_test)
score7 = mean_squared_error(Y7_pred,Y7_test)
mae7 = mean_absolute_error(Y7_pred,Y7_test)


# In[121]:


filtered_df7 = X.loc[:, [col for col in feature_filtered7 if col in X.columns]]

final_KNN_7 = KNeighborsRegressor(n_neighbors = 20)
X7_train, X7_test, Y7_train, Y7_test = train_test_split(filtered_df1, y, test_size=0.7, random_state=42)
final_KNN_7.fit(X7_train,Y7_train)
ypred7 = final_KNN_7.predict(X7_test)
score7_final = mean_squared_error(ypred7,Y7_test)
mae7_final = mean_absolute_error(ypred7,Y7_test)


# In[122]:


print(score7)
print(mae7)
print(score7_final)
print(mae7_final)


# # Dummy Regressor

# ## Model 1

# In[131]:


from sklearn.dummy import DummyRegressor
X_train, X_test, Y_train, Y_test = train_test_split(X, y_1, test_size=0.2, random_state=42)
dummy = DummyRegressor(strategy="mean").fit(X_train,Y_train)
ydummy = dummy.predict(X_test)
mse = mean_squared_error(ydummy,Y_test)
mae = mean_absolute_error(ydummy,Y_test)

print(mse)
print(mae)


# ## Model 2

# In[133]:


from sklearn.dummy import DummyRegressor
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X, y_2, test_size=0.2, random_state=42)
dummy = DummyRegressor(strategy="mean").fit(X2_train,Y2_train)
ydummy = dummy.predict(X2_test)
mse2 = mean_squared_error(ydummy,Y2_test)
mae2 = mean_absolute_error(ydummy,Y2_test)

print(mse2)
print(mae2)


# ## Model 3

# In[134]:


from sklearn.dummy import DummyRegressor
X3_train, X3_test, Y3_train, Y3_test = train_test_split(X, y_3, test_size=0.3, random_state=42)
dummy = DummyRegressor(strategy="mean").fit(X3_train,Y3_train)
ydummy = dummy.predict(X3_test)
mse3 = mean_squared_error(ydummy,Y3_test)
mae3 = mean_absolute_error(ydummy,Y3_test)

print(mse3)
print(mae3)


# ## Model 4

# In[135]:


from sklearn.dummy import DummyRegressor
X4_train, X4_test, Y4_train, Y4_test = train_test_split(X, y_4, test_size=0.4, random_state=42)
dummy = DummyRegressor(strategy="mean").fit(X4_train,Y4_train)
ydummy = dummy.predict(X4_test)
mse4 = mean_squared_error(ydummy,Y4_test)
mae4 = mean_absolute_error(ydummy,Y4_test)

print(mse4)
print(mae4)


# ## Model 5

# In[136]:


from sklearn.dummy import DummyRegressor
X5_train, X5_test, Y5_train, Y5_test = train_test_split(X, y_5, test_size=0.5, random_state=42)
dummy = DummyRegressor(strategy="mean").fit(X5_train,Y5_train)
ydummy = dummy.predict(X5_test)
mse5 = mean_squared_error(ydummy,Y5_test)
mae5 = mean_absolute_error(ydummy,Y5_test)


print(mse5)
print(mae5)


# ## Model 6

# In[137]:


from sklearn.dummy import DummyRegressor
X6_train, X6_test, Y6_train, Y6_test = train_test_split(X, y_6, test_size=0.6, random_state=42)
dummy = DummyRegressor(strategy="mean").fit(X6_train,Y6_train)
ydummy = dummy.predict(X6_test)
mse6 = mean_squared_error(ydummy,Y6_test)
mae6 = mean_absolute_error(ydummy,Y6_test)

print(mse6)
print(mae6)


# ## Model 7

# In[138]:


from sklearn.dummy import DummyRegressor
X7_train, X7_test, Y7_train, Y7_test = train_test_split(X, y_7, test_size=0.7, random_state=42)
dummy = DummyRegressor(strategy="mean").fit(X7_train,Y7_train)
ydummy = dummy.predict(X7_test)
mse7 = mean_squared_error(ydummy,Y7_test)
mae7 = mean_absolute_error(ydummy,Y7_test)

print(mse7)
print(mae7)


# In[ ]:




