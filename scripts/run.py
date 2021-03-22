#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
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


from implementations import *


# In[ ]:


Xtest = []
Xids = []

#kNN Baseline Movie
rmse, Xtest, Xids, preds_test, predictions = knn_baseline_movie(trainset, testset, ids, Xtest, Xids)


# In[ ]:


from data_helpers import create_csv_submission

OUTPUT_PATH = "../data/submission.csv"
create_csv_submission(ids, predictions, OUTPUT_PATH)
print("File submission.csv ready to be submitted !")

