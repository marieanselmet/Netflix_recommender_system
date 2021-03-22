from itertools import groupby
import numpy as np
#import pandas as pd
from surprise import accuracy
import scipy
import scipy.io
import scipy.sparse as sp

def calculate_mse(real_label, prediction):
    """calculate MSE."""
    t = real_label - prediction
    return 1.0 * t.dot(t.T)

def calculate_rmse(real_label, prediction):
    """calculate RMSE."""
    mse = calculate_mse(real_label, prediction)
    return np.sqrt(mse/len(real_label))

def baseline_global_mean(train, test, ids):
    """baseline method: use the global mean."""
    
    #Global mean
    global_mean = np.sum(train[np.nonzero(train)])/train.nnz  
    to_predict = test[np.nonzero(test)].todense()
    mse = calculate_mse(to_predict, global_mean)
    rmse = np.sqrt(mse / test.nnz)
    
    #Prediction
    predictions = []
    global_mean = int(round(global_mean))
    for i in range(len(ids[0])):
        predictions.append(global_mean)
    
    return rmse, predictions


def baseline_user_mean(train, test, ids):
    """baseline method: use the user means as the prediction."""
    #User means
    mse = 0
    _, num_users = train.shape
    user_means = np.zeros(num_users)
    
    for i in range(num_users):
        train_i = train[:,i]
        mean = np.mean(train_i[train_i.nonzero()])
        user_means[i] = mean
        
        test_i = test[:,i]
        to_predict = test_i[test_i.nonzero()].todense()
    
        mse += calculate_mse (to_predict, mean)
        
    rmse = np.sqrt(mse/test.nnz)
    
    #Prediction
    predictions = []
    for i in range(len(ids[0])):
        user = ids[1][i]
        mean = int(round(user_means[user-1]))
        predictions.append(mean)
        
    return rmse, predictions


def baseline_item_mean(train, test, ids):
    """baseline method: use item means as the prediction."""
    #Item means
    mse = 0
    num_items,_ = train.shape
    item_means = np.zeros(num_items)
    
    for i in range (num_items):
        train_i = train[i,:]
        mean = np.mean(train_i[train_i.nonzero()])
        item_means[i] = mean
        
        test_i = test[i,:]
        to_predict = test_i[test_i.nonzero()].todense()
    
        mse += calculate_mse (to_predict, mean)
        
    rmse = np.sqrt(mse/test.nnz)
    
    #Prediction
    predictions = []
    for i in range(len(ids[0])):
        item = ids[0][i]
        mean = int(round(item_means[item-1]))
        predictions.append(mean)
    
    return rmse, predictions
    
def group_by(data, index):
    """group list of list by a specific index."""
    sorted_data = sorted(data, key=lambda x: x[index])
    groupby_data = groupby(sorted_data, lambda x: x[index])
    return groupby_data

def build_index_groups(train):
    """build groups for nnz rows and cols."""
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    grouped_nz_train_byrow = group_by(nz_train, index=0)
    nz_row_colindices = [(g, np.array([v[1] for v in value]))
                         for g, value in grouped_nz_train_byrow]

    grouped_nz_train_bycol = group_by(nz_train, index=1)
    nz_col_rowindices = [(g, np.array([v[0] for v in value]))
                         for g, value in grouped_nz_train_bycol]
    return nz_train, nz_row_colindices, nz_col_rowindices

def init_MF(train, num_features):
    """init the parameter for matrix factorization.
        return:
            user_features: shape = num_features, num_user
            item_features: shape = num_features, num_item
    """
    num_item, num_user = train.get_shape()

    user_features = np.random.rand(num_features, num_user)
    item_features = np.random.rand(num_features, num_item)

    # start by item features.
    item_nnz = train.getnnz(axis=1) 
    item_sum = train.sum(axis=1)

    for ind in range(num_item):
        item_features[0, ind] = item_sum[ind, 0] / item_nnz[ind]
    
    return user_features, item_features

def compute_error(data, user_features, item_features, nz):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    mse = 0
    for row, col in nz:
        item_info = item_features[:, row]
        user_info = user_features[:, col]
        mse += (data[row, col] - user_info.T.dot(item_info)) ** 2 
    return np.sqrt(1.0 * mse / len(nz))

def compute_predictions(data, user_features, item_features, nz):
    """compute the prediction of nonzero elements."""
    predictions = []
    for row, col in nz:
        item_info = item_features[:, row]
        user_info = user_features[:, col]
        predictions.append(user_info.T.dot(item_info))       
    return predictions
        

def update_user_feature(train, item_features, lambda_user, nnz_items_per_user, nz_user_itemindices):
    """update user feature matrix."""
    """the best lambda is assumed to be nnz_items_per_user[user] * lambda_user"""
    num_user = nnz_items_per_user.shape[0]
    num_feature = item_features.shape[0]
    lambda_I = lambda_user * sp.eye(num_feature)
    updated_user_features = np.zeros((num_feature, num_user))

    for user, items in nz_user_itemindices:
        # extract the columns corresponding to the prediction for given item
        M = item_features[:, items]
        
        # update column of user features
        B = M @ train[items, user] # B = W.T.dot(X)
        A = M @ M.T + nnz_items_per_user[user] * lambda_I
        X = np.linalg.solve(A, B) #transposé par rapport au cours
        updated_user_features[:, user] = np.copy(X.T)
    return updated_user_features

def update_item_feature(train, user_features, lambda_item, nnz_users_per_item, nz_item_userindices):
    """update item feature matrix."""
    """the best lambda is assumed to be nnz_items_per_item[item] * lambda_item"""
    num_item = nnz_users_per_item.shape[0]
    num_feature = user_features.shape[0]
    lambda_I = lambda_item * sp.eye(num_feature)
    updated_item_features = np.zeros((num_feature, num_item))

    for item, users in nz_item_userindices:
        # extract the columns corresponding to the prediction for given user
        M = user_features[:, users]
        
        # update column of item features
        B = M @ train[item, users].T
        A = M @ M.T + nnz_users_per_item[item] * lambda_I
        X = np.linalg.solve(A, B)
        updated_item_features[:, item] = np.copy(X.T)
    return updated_item_features
