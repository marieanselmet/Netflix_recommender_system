#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import surprise
get_ipython().run_line_magic('load_ext', 'autoreload')


# In[ ]:


from data_helpers import *

#Load train and testset for the surprise models
file_path = "../data/data_surprise.csv"
trainset, testset = build_surprise_data(file_path)

#Loads ratings to predict
INPUT_PATH = "../data/sample_submission.csv"
ids = read_csv_sample(INPUT_PATH)


# In[ ]:


#Load train and testset for the custom models
ratings = load_data("../data/data_train.csv")
train, test = split_data(ratings, p_test = 0.1)
ratings.shape


# In[ ]:


from implementations import *


# In[ ]:


train.shape


# In[ ]:


Xtest = []
Xids = []

#Generate predictions with every method
rmse, Xtest, Xids, preds_test_kbm, preds_ids_kbm = knn_baseline_movie(trainset, testset, ids, Xtest, Xids)
rmse, Xtest, Xids, preds_test_kbu, preds_ids_kbu = knn_baseline_user(trainset, testset, ids, Xtest, Xids)
rmse, Xtest, Xids, preds_test_svd, preds_ids_svd = svd(trainset, testset, ids, Xtest, Xids)
rmse, Xtest, Xids, preds_test_als, preds_ids_als = matrix_factorization_als2(train, test, ids, Xtest, Xids)
rmse, Xtest, Xids, preds_test_sgd, preds_ids_sgd = matrix_factorization_sgd1(train.T, test.T, ids, Xtest, Xids)


# In[ ]:


from sklearn.linear_model import LinearRegression
def blend(preds_test, preds_ids, testset):
    """
    Linear regression that finds the optimal weights of each model
    Argument : preds_test, predicted ratings for the known test set
               preds_ids, predicted ratings for the unknown set
               testset, the testset
    Return : estimations, the final predictions
             weights, coefficients associated to each model
    """
    print('Blending')
    
    #Known ratings of testset
    y_test = [rating for (_,_,rating) in testset]
    
    #Ridge Regression
    linreg = Ridge(alpha=0.1, normalize=True)
    
    #Fit between predicted and know ratings of testset
    linreg.fit(preds_test.T, y_test)
    weights = linreg.coef_
    
    #Predict unknown ratings
    predictions = np.clip(linreg.predict(preds_ids.T), 1, 5)
    
    print(weights, end='\n\n')
    
    #RMSE of regression
    print('Test RMSE: %f' % calculate_rmse(y_test, linreg.predict(preds_test.T)))
    
    #Rounding-off predictions
    estimations = np.zeros(len(predictions))
    for j, pred in enumerate(predictions):
        estimations[j] = round(pred)
        
    return estimations, weights


# In[ ]:


#Blending
predictions, weights = blend(np.array(Xtest), np.array(Xids), testset)


# In[ ]:


len(predictions)


# In[ ]:


from data_helpers import create_csv_submission

OUTPUT_PATH = "../data/submission.csv"
create_csv_submission(ids, predictions, OUTPUT_PATH)
print("File submission.csv ready to be submitted !")


# In[ ]:




