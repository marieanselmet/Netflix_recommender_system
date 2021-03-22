from surprise import NormalPredictor
from proj2_helpers import *
from surprise import accuracy
from surprise import BaselineOnly
from surprise import SVD
from surprise import SVDpp
from surprise import SlopeOne
from surprise import KNNBaseline
from surprise import KNNBasic
from surprise import NMF
from surprise import CoClustering
from sklearn.linear_model import Ridge
from surprise import KNNWithMeans
import scipy
import scipy.io
import scipy.sparse as sp

def global_mean(train, test, ids):
    """baseline method: use the global mean.
    Argument: train, the trainset
              test, the testset
              ids, the unknown ratings
    """
    print ('Global Mean')
    #Global mean
    global_mean = np.sum(train[np.nonzero(train)])/train.nnz
    #Non-zero values in testset
    to_predict = test[np.nonzero(test)].todense()
    
    #RMSE on train
    mse_train = calculate_mse(train[np.nonzero(train)].todense(), global_mean)
    rmse_train = np.sqrt(mse_train/train.nnz)
    print('   Training RMSE: ', np.squeeze(rmse_train))
    
    #RMSE on test
    mse = calculate_mse(to_predict, global_mean)
    rmse = np.sqrt(mse/test.nnz)
    print('   Test RMSE: ', np.squeeze(rmse))
    
    #Prediction
    predictions = []
    global_mean = int(round(global_mean))
    for i in range(len(ids[0])):
        predictions.append(global_mean)
    
    return rmse, predictions

def user_mean(train, test, ids):
    """baseline method: use the user means as the prediction.
    Argument: train, the trainset
              test, the testset
              ids, the unknown ratings
    """
    print('User Mean')
    #User means
    mse = 0
    mse_train = 0
    _, num_users = train.shape
    user_means = np.zeros(num_users)
    
    for i in range(num_users):
        train_i = train[:,i]
        #User mean
        mean = np.mean(train_i[train_i.nonzero()])
        user_means[i] = mean
        
        test_i = test[:,i]
        to_predict = test_i[test_i.nonzero()].todense()
        
        mse_train += calculate_mse(train_i[train_i.nonzero()].todense(), mean)
        mse += calculate_mse (to_predict, mean)
    
    #RMSE on train
    rmse_train = np.sqrt(mse_train/train.nnz)
    print('   Training RMSE: ', np.squeeze(rmse_train))
    
    #RMSE on test
    rmse = np.sqrt(mse/test.nnz)
    print('   Test RMSE: ', np.squeeze(rmse))
        
    #Prediction
    predictions = []
    for i in range(len(ids[0])):
        user = ids[1][i]
        mean = int(round(user_means[user-1]))
        predictions.append(mean)
        
    return rmse, predictions


def item_mean(train, test, ids):
    """baseline method: use item means as the prediction.
    Argument: train, the trainset
              test, the testset
              ids, the unknown ratings
    """
    print ('Item Mean')
    #Item means
    mse = 0
    num_items,_ = train.shape
    item_means = np.zeros(num_items)
    
    for i in range (num_items):
        #Item mean
        train_i = train[i,:]
        mean = np.mean(train_i[train_i.nonzero()])
        item_means[i] = mean
        test_i = test[i,:]
        to_predict = test_i[test_i.nonzero()].todense()
        
        mse_train += calculate_mse(train_i[train_i.nonzero()].todense(), mean)
        mse += calculate_mse (to_predict, mean)
    
    #RMSE on train
    rmse_train = np.sqrt(mse_train/train.nnz)
    print('   Training RMSE: ', np.squeeze(rmse_train))
    
    #RMSE on test
    rmse = np.sqrt(mse/test.nnz)
    print('   Test RMSE: ', np.squeeze(rmse))
   
    #Ids Prediction (unknown ratings)
    predictions = []
    for i in range(len(ids[0])):
        item = ids[0][i]
        mean = int(round(item_means[item-1]))
        predictions.append(mean)

    return rmse, predictions


def normal_predictor(train, test, ids, Xtest, Xids):
    """
    Generates predictions according to a normal distribution estimated from the training set
    Argument : train, the trainset
               test, the testset
               ids, unknown ratings
               Xtest, predicted ratings for testset, to be used for final blending
               Xids, predicted ratings for unknown ratings, to be used for final blending
    """
    print('Normal Predictor')
    algo = NormalPredictor()
        
    #Train algorithm on training set
    algo.fit(train)
    
    #Predict on train and compute RMSE
    predictions = algo.test(train.build_testset())
    print('   Training RMSE: ', accuracy.rmse(predictions, verbose=False))
    
    #Predict on test and compute RMSE
    predictions = algo.test(test)
    rmse = accuracy.rmse(predictions, verbose=False)
    print('   Test RMSE: ', rmse)
    
    preds_test = np.zeros(len(predictions))
    for j, pred in enumerate(predictions):
        preds_test[j] = pred.est
    
    #Predict unknown ratings
    preds_ids = []
    for i in range(len(ids[0])):
        pred = algo.predict(str(ids[0][i]), str(ids[1][i]))
        preds_ids.append(pred.est)

    Xtest.append(preds_test)
    Xids.append(preds_ids)
    return rmse, Xtest, Xids, preds_test, preds_ids


def baseline_only(train, test, ids, Xtest, Xids):
    """
    Combines user and item mean with user and item biases
    Argument : train, the trainset
               test, the testset
               ids, unknown ratings
               Xtest, predicted ratings for testset, to be used for final blending
               Xids, predicted ratings for unknown ratings, to be used for final blending
    """
    print('Baseline Only')
    bsl_options = {'method': 'als',
               'n_epochs': 100,
               'reg_u': 15,
               'reg_i': 0.01
               }

    algo = BaselineOnly(bsl_options=bsl_options, verbose = False)
        
    #Train algorithm on training set
    algo.fit(train)
    
    #Predict on train and compute RMSE
    predictions = algo.test(train.build_testset())
    print('   Training RMSE: ', accuracy.rmse(predictions, verbose=False))
    
    #Predict on test and compute RMSE
    predictions = algo.test(test)
    rmse = accuracy.rmse(predictions, verbose=False)
    print('   Test RMSE: ', rmse)
    
    preds_test = np.zeros(len(predictions))
    for j, pred in enumerate(predictions):
        preds_test[j] = pred.est
    
    #Predict unknown ratings
    preds_ids = []
    for i in range(len(ids[0])):
        pred = algo.predict(str(ids[0][i]), str(ids[1][i]))
        preds_ids.append(pred.est)

    Xtest.append(preds_test)
    Xids.append(preds_ids)
    return rmse, Xtest, Xids, preds_test, preds_ids

def knn_baseline_user(train, test, ids, Xtest, Xids):
    """
    nearest neighbour approach using the user baseline
    Argument : train, the trainset
               test, the testset
               ids, unknown ratings
               Xtest, predicted ratings for testset, to be used for final blending
               Xids, predicted ratings for unknown ratings, to be used for final blending
    """
    print('kNN Baseline User')
    bsl_option = {'method': 'als',
               'n_epochs': 10,
               'reg_u': 15,
               'reg_i': 0.01
               }
    sim_option = {'name': 'pearson_baseline',
                              'min_support': 1,
                              'user_based': True }
    
    algo = KNNBaseline(k = 400, bsl_options= bsl_option, sim_options= sim_option, verbose = False)
        
    #Train algorithm on training set
    algo.fit(train)
    
    #Predict on train and compute RMSE
    predictions = algo.test(train.build_testset())
    print('   Training RMSE: ', accuracy.rmse(predictions, verbose=False))
    
    #Predict on test and compute RMSE
    predictions = algo.test(test)
    rmse = accuracy.rmse(predictions, verbose=False)
    print('   Test RMSE: ', rmse)
    
    preds_test = np.zeros(len(predictions))
    for j, pred in enumerate(predictions):
        preds_test[j] = pred.est
    
    #Predict unknown ratings
    preds_ids = []
    for i in range(len(ids[0])):
        pred = algo.predict(str(ids[0][i]), str(ids[1][i]))
        preds_ids.append(pred.est)

    Xtest.append(preds_test)
    Xids.append(preds_ids)
    return rmse, Xtest, Xids, preds_test, preds_ids

def knn_baseline_movie(train, test, ids, Xtest, Xids):
    """
    nearest neighbour approach using the movie baseline
    Argument : train, the trainset
               test, the testset
               ids, unknown ratings
               Xtest, predicted ratings for testset, to be used for final blending
               Xids, predicted ratings for unknown ratings, to be used for final blending
    """
    
    print ('kNN Baseline Movie')
    bsl_option = {'method': 'als',
               'n_epochs': 100,
               'reg_u': 15,
               'reg_i': 0.01
               }

    sim_option = {'name': 'pearson_baseline',
                              'min_support': 1,
                              'user_based': False }

    algo = KNNBaseline(k = 100, bsl_options= bsl_option, sim_options= sim_option, verbose = False)
        
    #Train algorithm on training set
    algo.fit(train)
    
    #Predict on train and compute RMSE
    predictions = algo.test(train.build_testset())
    print('   Training RMSE: ', accuracy.rmse(predictions, verbose=False))
    
    #Predict on test and compute RMSE
    predictions = algo.test(test)
    rmse = accuracy.rmse(predictions, verbose=False)
    print('   Test RMSE: ', rmse)
    
    preds_test = np.zeros(len(predictions))
    for j, pred in enumerate(predictions):
        preds_test[j] = pred.est
    
    #Predict unknown ratings
    preds_ids = []
    for i in range(len(ids[0])):
        pred = algo.predict(str(ids[0][i]), str(ids[1][i]))
        preds_ids.append(pred.est)

    Xtest.append(preds_test)
    Xids.append(preds_ids)
    return rmse, Xtest, Xids, preds_test, preds_ids

def svd(train, test, ids, Xtest, Xids):
    """
    Matrix-factorization taking biases into account
    Argument : train, the trainset
               test, the testset
               ids, unknown ratings
               Xtest, predicted ratings for testset, to be used for final blending
               Xids, predicted ratings for unknown ratings, to be used for final blending
    """
    print ('SVD')
    algo = SVD(n_factors=100, n_epochs=40, lr_all=0.0015, reg_all= 0.05, biased = True, random_state = 15)
        
    #Train algorithm on training set
    algo.fit(train)
    
    #Predict on train and compute RMSE
    predictions = algo.test(train.build_testset())
    print('   Training RMSE: ', accuracy.rmse(predictions, verbose=False))
    
    #Predict on test and compute RMSE
    predictions = algo.test(test)
    rmse = accuracy.rmse(predictions, verbose=False)
    print('   Test RMSE: ', rmse)
    
    preds_test = np.zeros(len(predictions))
    for j, pred in enumerate(predictions):
        preds_test[j] = pred.est
    
    #Predict unknown ratings
    preds_ids = []
    for i in range(len(ids[0])):
        pred = algo.predict(str(ids[0][i]), str(ids[1][i]))
        preds_ids.append(pred.est)

    Xtest.append(preds_test)
    Xids.append(preds_ids)
    return rmse, Xtest, Xids, preds_test, preds_ids


def svdpp(train, test, ids, Xtest, Xids):
    """
    Extension of svd taking the implicit ratings into account
    Argument : train, the trainset
               test, the testset
               ids, unknown ratings
               Xtest, predicted ratings for testset, to be used for final blending
               Xids, predicted ratings for unknown ratings, to be used for final blending
    """
    print ('SVD++')
    algo = SVDpp(n_factors=100, n_epochs=10, lr_all=0.0015, reg_all= 0.05, random_state = 15)
                
        
    #Train algorithm on training set
    algo.fit(train)
    
    #Predict on train and compute RMSE
    predictions = algo.test(train.build_testset())
    print('   Training RMSE: ', accuracy.rmse(predictions, verbose=False))
    
    #Predict on test and compute RMSE
    predictions = algo.test(test)
    rmse = accuracy.rmse(predictions, verbose=False)
    print('   Test RMSE: ', rmse)
    
    preds_test = np.zeros(len(predictions))
    for j, pred in enumerate(predictions):
        preds_test[j] = pred.est
    
    #Predict unknown ratings
    preds_ids = []
    for i in range(len(ids[0])):
        pred = algo.predict(str(ids[0][i]), str(ids[1][i]))
        preds_ids.append(pred.est)

    Xtest.append(preds_test)
    Xids.append(preds_ids)
    return rmse, Xtest, Xids, preds_test, preds_ids

def slopeone(train, test,  ids, Xtest, Xids):
    """
    Item based algorithm, reduces overfitting
    Argument : train, the trainset
               test, the testset
               ids, unknown ratings
               Xtest, predicted ratings for testset, to be used for final blending
               Xids, predicted ratings for unknown ratings, to be used for final blending
    """
    
    print('SlopeOne')
    algo = SlopeOne()
    
        
    #Train algorithm on training set
    algo.fit(train)
    
    #Predict on train and compute RMSE
    predictions = algo.test(train.build_testset())
    print('   Training RMSE: ', accuracy.rmse(predictions, verbose=False))
    
    #Predict on test and compute RMSE
    predictions = algo.test(test)
    rmse = accuracy.rmse(predictions, verbose=False)
    print('   Test RMSE: ', rmse)
    
    preds_test = np.zeros(len(predictions))
    for j, pred in enumerate(predictions):
        preds_test[j] = pred.est
    
    #Predict unknown ratings
    preds_ids = []
    for i in range(len(ids[0])):
        pred = algo.predict(str(ids[0][i]), str(ids[1][i]))
        preds_ids.append(pred.est)

    Xtest.append(preds_test)
    Xids.append(preds_ids)
    return rmse, Xtest, Xids, preds_test, preds_ids

def nmf(train, test,  ids, Xtest, Xids):
    """
    Non Negative Matrix Factorization
    Argument : train, the trainset
               test, the testset
               ids, unknown ratings
               Xtest, predicted ratings for testset, to be used for final blending
               Xids, predicted ratings for unknown ratings, to be used for final blending
    """
    print('NMF')
    algo = NMF(n_factors=20, n_epochs=50, random_state=15, reg_pu=0.5, reg_qi=0.05)
        
    #Train algorithm on training set
    algo.fit(train)
    
    #Predict on train and compute RMSE
    predictions = algo.test(train.build_testset())
    print('   Training RMSE: ', accuracy.rmse(predictions, verbose=False))
    
    #Predict on test and compute RMSE
    predictions = algo.test(test)
    rmse = accuracy.rmse(predictions, verbose=False)
    print('   Test RMSE: ', rmse)
    
    preds_test = np.zeros(len(predictions))
    for j, pred in enumerate(predictions):
        preds_test[j] = pred.est
    
    #Predict unknown ratings
    preds_ids = []
    for i in range(len(ids[0])):
        pred = algo.predict(str(ids[0][i]), str(ids[1][i]))
        preds_ids.append(pred.est)

    Xtest.append(preds_test)
    Xids.append(preds_ids)
    return rmse, Xtest, Xids, preds_test, preds_ids
    
def co_clustering(train, test,  ids, Xtest, Xids):
    """
    Co-clustering algorithm, users and items assigned to clusters and co_clusters
    Argument : train, the trainset
               test, the testset
               ids, unknown ratings
               Xtest, predicted ratings for testset, to be used for final blending
               Xids, predicted ratings for unknown ratings, to be used for final blending
    """
    print('Co-clustering')
    algo = CoClustering(n_cltr_u=1, n_cltr_i=1, n_epochs=50, random_state = 15)
    
    #Train algorithm on training set
    algo.fit(train)
    
    #Predict on train and compute RMSE
    predictions = algo.test(train.build_testset())
    print('   Training RMSE: ', accuracy.rmse(predictions, verbose=False))
    
    #Predict on test and compute RMSE
    predictions = algo.test(test)
    rmse = accuracy.rmse(predictions, verbose=False)
    print('   Test RMSE: ', rmse)
    
    preds_test = np.zeros(len(predictions))
    for j, pred in enumerate(predictions):
        preds_test[j] = pred.est
    
    #Predict unknown ratings
    preds_ids = []
    for i in range(len(ids[0])):
        pred = algo.predict(str(ids[0][i]), str(ids[1][i]))
        preds_ids.append(pred.est)

    Xtest.append(preds_test)
    Xids.append(preds_ids)
    return rmse, Xtest, Xids, preds_test, preds_ids

def knn_centered_user(train, test,  ids, Xtest, Xids):
    """
    kNN approach taking into account the mean ratings of each user
    Argument : train, the trainset
               test, the testset
               ids, unknown ratings
               Xtest, predicted ratings for testset, to be used for final blending
               Xids, predicted ratings for unknown ratings, to be used for final blending
    """
    print('Centered kNN User')
    algo = KNNWithMeans(k= 200, name = 'pearson_baseline', min_support =5, user_based = True, shrinkage =120 )
    
    #Train algorithm on training set
    algo.fit(train)
    
    #Predict on train and compute RMSE
    predictions = algo.test(train.build_testset())
    print('   Training RMSE: ', accuracy.rmse(predictions, verbose=False))
    
    #Predict on test and compute RMSE
    predictions = algo.test(test)
    rmse = accuracy.rmse(predictions, verbose=False)
    print('   Test RMSE: ', rmse)
    
    preds_test = np.zeros(len(predictions))
    for j, pred in enumerate(predictions):
        preds_test[j] = pred.est
    
    #Predict unknown ratings
    preds_ids = []
    for i in range(len(ids[0])):
        pred = algo.predict(str(ids[0][i]), str(ids[1][i]))
        preds_ids.append(pred.est)

    Xtest.append(preds_test)
    Xids.append(preds_ids)
    return rmse, Xtest, Xids, preds_test, preds_ids

def knn_centered_movie(train, test,  ids, Xtest, Xids):
    """
    kNN approach taking into account the mean ratings of each movie
    Argument : train, the trainset
               test, the testset
               ids, unknown ratings
               Xtest, predicted ratings for testset, to be used for final blending
               Xids, predicted ratings for unknown ratings, to be used for final blending
    """
    print('Centered kNN Movie')
    algo = KNNWithMeans(k= 65, name = 'msd', min_support =1, user_based = False)
    
    #Train algorithm on training set
    algo.fit(train)
    
    #Predict on train and compute RMSE
    predictions = algo.test(train.build_testset())
    print('   Training RMSE: ', accuracy.rmse(predictions, verbose=False))
    
    #Predict on test and compute RMSE
    predictions = algo.test(test)
    rmse = accuracy.rmse(predictions, verbose=False)
    print('   Test RMSE: ', rmse)
    
    preds_test = np.zeros(len(predictions))
    for j, pred in enumerate(predictions):
        preds_test[j] = pred.est
    
    #Predict unknown ratings
    preds_ids = []
    for i in range(len(ids[0])):
        pred = algo.predict(str(ids[0][i]), str(ids[1][i]))
        preds_ids.append(pred.est)

    Xtest.append(preds_test)
    Xids.append(preds_ids)
    return rmse, Xtest, Xids, preds_test, preds_ids

def knn_basic_user(train, test,  ids, Xtest, Xids):
    """
    kNN basic approach on users
    Argument : train, the trainset
               test, the testset
               ids, unknown ratings
               Xtest, predicted ratings for testset, to be used for final blending
               Xids, predicted ratings for unknown ratings, to be used for final blending
    """
    
    print('kNN Basic User')
    algo = KNNBasic(k=250, name = 'msd', min_support = 2, user_based = True, verbose = False)
    
    #Train algorithm on training set
    algo.fit(train)
    
    #Predict on train and compute RMSE
    predictions = algo.test(train.build_testset())
    print('   Training RMSE: ', accuracy.rmse(predictions, verbose=False))
    
    #Predict on test and compute RMSE
    predictions = algo.test(test)
    rmse = accuracy.rmse(predictions, verbose=False)
    print('   Test RMSE: ', rmse)
    
    preds_test = np.zeros(len(predictions))
    for j, pred in enumerate(predictions):
        preds_test[j] = pred.est
    
    #Predict unknown ratings
    preds_ids = []
    for i in range(len(ids[0])):
        pred = algo.predict(str(ids[0][i]), str(ids[1][i]))
        preds_ids.append(pred.est)

    Xtest.append(preds_test)
    Xids.append(preds_ids)
    return rmse, Xtest, Xids, preds_test, preds_ids

def knn_basic_movie(train, test,  ids, Xtest, Xids):
    """
    kNN basic approach on movies
    Argument : train, the trainset
               test, the testset
               ids, unknown ratings
               Xtest, predicted ratings for testset, to be used for final blending
               Xids, predicted ratings for unknown ratings, to be used for final blending
    """
    
    print('kNN Basic Movie')
    algo = KNNBasic(k=21, name = 'msd', min_support = 2, user_based = False, verbose = False)
    
    #Train algorithm on training set
    algo.fit(train)
    
    #Predict on train and compute RMSE
    predictions = algo.test(train.build_testset())
    print('   Training RMSE: ', accuracy.rmse(predictions, verbose=False))
    
    #Predict on test and compute RMSE
    predictions = algo.test(test)
    rmse = accuracy.rmse(predictions, verbose=False)
    print('   Test RMSE: ', rmse)
    
    preds_test = np.zeros(len(predictions))
    for j, pred in enumerate(predictions):
        preds_test[j] = pred.est
    
    #Predict unknown ratings
    preds_ids = []
    for i in range(len(ids[0])):
        pred = algo.predict(str(ids[0][i]), str(ids[1][i]))
        preds_ids.append(pred.est)

    Xtest.append(preds_test)
    Xids.append(preds_ids)
    return rmse, Xtest, Xids, preds_test, preds_ids

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
    linreg = Ridge(alpha=0.1, fit_intercept=False)
    
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

def matrix_factorization_als(train, test, ids, Xtest, Xids):
    """Alternating Least Squares (ALS) algorithm.
    Argument : train, the trainset
               test, the testset
               ids, unknown ratings
               Xtest, predicted ratings for testset, to be used for final blending
               Xids, predicted ratings for unknown ratings, to be used for final blending
    """
    # define parameters
    num_features = 20
    lambda_user = 0.08
    lambda_item = 0.1
    stop_criterion = 1e-4
    change = 1
    error_list = [0, 0]

    # set seed
    np.random.seed(988)

    # init ALS
    user_features, item_features = init_MF(train, num_features)
    
    # get the number of non-zero ratings for each user and item
    nnz_items_per_user, nnz_users_per_item = train.getnnz(axis=0), train.getnnz(axis=1)
    
    # group the indices by row or column index
    nz_train, nz_item_userindices, nz_user_itemindices = build_index_groups(train)

    # run ALS
    print("start the ALS algorithm...")
    while change > stop_criterion:
        # update user feature & item feature
        user_features = update_user_feature(train, item_features, lambda_user, nnz_items_per_user, nz_user_itemindices)
        item_features = update_item_feature(train, user_features, lambda_item, nnz_users_per_item, nz_item_userindices)

        error = compute_error(train, user_features, item_features, nz_train)
        
        error_list.append(error)
        change = np.fabs(error_list[-1] - error_list[-2])
        
    print("Training RMSE: {}.".format(error))
    # evaluate the test error
    nnz_row, nnz_col = test.nonzero()
    nnz_test = list(zip(nnz_row, nnz_col))
    rmse = compute_error(test, user_features, item_features, nnz_test)
    print("Test RMSE: {v}.".format(v=rmse))
    
    predictions_matrix = user_features.T @ item_features
    
    #Predict unknown ratings
    preds_ids = []
    for i in range(len(ids[0])):
        user = ids[0][i]
        item = ids[1][i]
        rating = round(predictions_matrix[item-1, user-1])
        preds_ids.append(rating)

    preds_ids = np.clip(preds_ids, 1, 5)
    Xids.append(preds_ids)
    
    #Predict test ratings (known)
    preds_test = compute_predictions(test, user_features, item_features, nnz_test)
    preds_test = np.clip(preds_test, 1, 5)
    Xtest.append(preds_test)
    return rmse, Xtest, Xids, preds_test, preds_ids

def matrix_factorization_sgd(train, test, ids, Xtest, Xids):
    """matrix factorization by SGD.
    Argument : train, the trainset
               test, the testset
               ids, unknown ratings
               Xtest, predicted ratings for testset, to be used for final blending
               Xids, predicted ratings for unknown ratings, to be used for final blending
    """
    # define parameters   
    gamma = 0.06
    num_features = 20   # K in the lecture notes
    lambda_user = 0.08
    lambda_item = 0.1
    num_epochs = 30
    errors = [0]
    
    # set seed
    np.random.seed(988)

    # init matrix
    user_features, item_features = init_MF(train, num_features)
    
    # find the non-zero ratings indices 
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    print("learn the matrix factorization using SGD...")
    for it in range(num_epochs):        
        # shuffle the training rating indices
        np.random.shuffle(nz_train) #changes the position of items in a list
        
        # decrease step size
        gamma /= 1.2
        
        for d, n in nz_train:
            # select W_d and Z_n
            item_info = item_features[:, d]
            user_info = user_features[:, n]
            
            #calculate the prediction error
            err = train[d, n] - user_info.T.dot(item_info)
    
            # calculate the gradient and update
            item_features[:, d] += gamma * (err * user_info - lambda_item * item_info)
            user_features[:, n] += gamma * (err * item_info - lambda_user * user_info)

        rmse = compute_error(train, user_features, item_features, nz_train)
        print("iter: {}, Train RMSE: {}.".format(it, rmse))
        
        errors.append(rmse)
    print("Training RMSE: {}".format(rmse))    
    # evaluate the test error
    rmse = compute_error(test, user_features, item_features, nnz_test)
    print("Test RMSE: {}.".format(rmse))
    predictions_matrix = user_features.T @ item_features
    preds_ids = []
    
    #Predict unknown ratings
    for i in range(len(ids[0])):
        user = ids[0][i]
        item = ids[1][i]
        rating = round(predictions_matrix[item-1, user-1])
        preds_ids.append(rating)

    preds_ids = np.clip(preds_ids, 1, 5)
    Xids.append(preds_ids)
    
    #Predict test ratings
    preds_test = compute_predictions(test, user_features, item_features, nnz_test)
    preds_test = np.clip(preds_test, 1, 5)
    Xtest.append(preds_test)
    
    return rmse, Xtest, Xids, preds_test, preds_ids