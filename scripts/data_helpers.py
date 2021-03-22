import csv
import numpy as np
import scipy.sparse as sp
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split

def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()

def deal_line(line):
    pos, rating = line.split(',')
    row, col = pos.split("_")
    row = row.replace("r", "")
    col = col.replace("c", "")
    return int(row), int(col), float(rating)

def preprocess_data(data):
    """preprocessing the text data, conversion to numerical array format."""
    def statistics(data):
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    # parse each line
    data = [deal_line(line) for line in data]

    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)
    print("number of users: {}, number of items: {}".format(max_row, max_col))

    # build rating matrix.
    ratings = sp.lil_matrix((max_row, max_col))
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
    return ratings.T


def load_data(path_dataset):
    """Load data in text format, one rating per line, as in the kaggle competition."""
    data = read_txt(path_dataset)[1:]
    return preprocess_data(data)


def read_csv_sample(path):
    """
    Reads the sample_submission file and extracts the couples (item, user) for which the rating has to be predicted.
    Argument: name (string name of .csv input file)
    """
    
    users = []
    items = []
    
    data = read_txt(path)[1:]
    
    for line in data:
        user, item, _ = deal_line(line)
        users.append(user)
        items.append(item)
     
    return [users, items]


def create_csv_submission(ids, predictions, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (
               ratings (predicted ratings)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w', newline='') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2, r3 in zip(ids[0], ids[1], predictions):
            writer.writerow({'Id':'r' + str(r1) + '_c' + str(r2),'Prediction':r3})

            
def build_surprise_data(path):
    """
    Loads the training set for it to be usable by the surprise library
    Argument: path (path of the file)
    """
    reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
    data = Dataset.load_from_file(path, reader=reader)
    trainset, testset = train_test_split(data, test_size=.1)
    return trainset, testset


def split_data(ratings, p_test=0.1):
    """split the ratings to training data and test data.
    Argument: ratings, original data set
              p_test, the proportion given to test set
    """
    # set seed
    np.random.seed(988) 
    
    # init
    num_rows, num_cols = ratings.shape
    train = sp.lil_matrix((num_rows, num_cols))
    test = sp.lil_matrix((num_rows, num_cols))
    
    nz_items, nz_users = ratings.nonzero() #return the indices of the elements that are non-zero
    
    # split the data
    for user in set(nz_users): 
        #for each colum (user), we chose p_test of movies for the test set, the remainder is for training set
        #randomly select a subset of ratings
        row, col = ratings[:, user].nonzero()
        selects = np.random.choice(row, size=int(len(row) * p_test)) #generates a random sample from a given 1-D array
        residual = list(set(row) - set(selects))

        # add to train set
        for res in residual:
            train[res, user] = ratings[res, user]

        # add to test set
        for sel in selects:
            test[sel, user] = ratings[sel, user]
    
    return train, test