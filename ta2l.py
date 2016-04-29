import os
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from scipy.stats import entropy
from scipy.stats import gaussian_kde
from scipy.stats import uniform

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import chi2
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM


def main():
    kl_archive = kl_init()

    kl_arr = kl_archive[-1]
    kl_arr = list(sorted(kl_arr))#[9 * len(kl_arr) // 10:]

    left = - int(max(len(kl_arr), len(kl_arr)) * 0.05)
    right = max(len(kl_arr), len(kl_arr)) - left
    bottom = kl_arr[0] - kl_arr[-1]/20
    top = kl_arr[-1] * 1.05
    plt.axis([left, right, bottom, top])
    ax = plt.gca()
    ax.set_autoscale_on(False)
    plt.xlabel("Feature Index")
    plt.ylabel("KL Divergence")
    plt.plot(kl_arr)
    plt.show()


def kl_init():
    archfile = 'kl_archive.pckl'
    if os.path.isfile(archfile):
        print("== Reloading KL Divergence Data ==")
        kl_archive = pickle.load(open(archfile, 'rb'))
    else:
        archive = init()
        X_train = archive[0]
        y_train = archive[1]
        X_test = archive[2]
        c_cols, d_cols = split_data_types(X_train, X_test)

        print("== Computing KL Divergence Data ==")
        kl_arr = kl_diverge(X_train, y_train, c_cols, d_cols)
        kl_archive = archive + [c_cols, d_cols, kl_arr]
        pickle.dump(kl_archive, open(archfile, 'wb'))

    return kl_archive

def init():
    archfile = 'archive.pckl'
    if os.path.isfile(archfile):
        print("== Reloading Data ==")
        archive = pickle.load(open(archfile, 'rb'))
    else:
        print("== Loading Data ==")
        X_train, y_train, X_test, id_test = load_data()
        archive = [
            X_train,
            y_train,
            X_test,
            id_test
        ]
        pickle.dump(archive, open(archfile, 'wb'))

    return archive

def load_data():
    # load data
    df_train = pd.read_csv('input/train.csv')
    df_test = pd.read_csv('input/test.csv')

    # remove constant columns
    remove = []
    for col in df_train.columns:
        if df_train[col].std() == 0:
            remove.append(col)

    df_train.drop(remove, axis=1, inplace=True)
    df_test.drop(remove, axis=1, inplace=True)

    # remove duplicated columns
    remove = []
    c = df_train.columns
    for i in range(len(c)-1):
        v = df_train[c[i]].values
        for j in range(i+1,len(c)):
            if np.array_equal(v,df_train[c[j]].values):
                remove.append(c[j])

    df_train.drop(remove, axis=1, inplace=True)
    df_test.drop(remove, axis=1, inplace=True)

    y_train = df_train['TARGET'].values
    X_train = df_train.drop(['ID','TARGET'], axis=1).values

    id_test = df_test['ID']
    X_test = df_test.drop(['ID'], axis=1).values

    return X_train, y_train, X_test, id_test

def split_data_types(X_train, X_test):
    print("== Splitting Continuous and Discrete ==")

    # Find continuous columns
    col_is_cont = [False]*X_train.shape[1]
    for c in range(X_train.shape[1]):
        for i in range(X_train.shape[0]):
            if int(X_train[i, c]) != X_train[i, c]:
                col_is_cont[c] = True
                break
    for c in range(X_test.shape[1]):
        if col_is_cont[c]:
            continue
        for i in range(X_test.shape[0]):
            if int(X_test[i, c]) != X_test[i, c]:
                col_is_cont[c] = True
                break

    c_cols = set(filter(lambda i: col_is_cont[i], range(len(col_is_cont))))
    d_cols = set(range(len(col_is_cont))) - c_cols

    return c_cols, d_cols

def preprocess(X_train, y_train, X_test):
    print("== Preprocessing Data ==")
    X_merge = np.concatenate((X_train, X_test), axis=0)
    sep = X_train.shape[0]

    # Scale features to range [0, 1]
    scale = MinMaxScaler()
    X_merge = scale.fit_transform(X_merge)

    X_train = X_merge[:sep]
    X_test = X_merge[sep:]

    # Choose top features as ranked by chi squared test
    gus = GenericUnivariateSelect(score_func=chi2, mode="k_best", param=306)
    gus.fit(X_train, y_train)
    X_train = gus.transform(X_train)
    X_test = gus.transform(X_test)

    return X_train, X_test

def kl_diverge(X_train, y_train, c_cols, d_cols):
    # Separate data matrix into Class 0 and Class 1 matrices
    num_0 = sum(map(lambda y: not y, y_train))
    num_1 = sum(y_train)
    X_0 = np.ndarray(shape=(num_0, X_train.shape[1]), dtype=np.float64)
    X_1 = np.ndarray(shape=(num_1, X_train.shape[1]), dtype=np.float64)

    idx_0 = 0
    idx_1 = 0
    for i in range(y_train.shape[0]):
        if not y_train[i]:
            X_0[idx_0, :] = X_train[i, :]
            idx_0 += 1
        else:
            X_1[idx_1, :] = X_train[i, :]
            idx_1 += 1

    kl_arr = [0] * X_train.shape[1]

    print("Computing KLD for continuous columns...")
    c_count = 0
    for i in c_cols:
        print("%d of %d..." % (c_count, len(c_cols)))
        # Get column
        col_0 = X_0[:,i]
        col_1 = X_1[:,i]
        col_0 = list(sorted(map(lambda r: r[i], X_0)))
        col_1 = list(sorted(map(lambda r: r[i], X_1)))

        # Determine 1000 values to compare
        left = min(col_0[0], col_1[0])
        right = max(col_0[-1], col_1[-1])
        domain = np.linspace(left, right, 1000)

        # If a column is always c for a class, then gaussian_kde() fails
        # In such cases, use a uniform distribution for the class
        elem_0 = set(col_0)
        elem_1 = set(col_1)
        if len(elem_0) == 1:
            pk = list(map(lambda x: 1 if x in elem_0 else 0, domain))
        else:
            dist0 = gaussian_kde(col_0)
            pk = dist0.pdf(domain)

        if len(elem_1) == 1:
            qk = list(map(lambda x: 1 if x in elem_1 else 0, domain))
        else:
            dist1 = gaussian_kde(col_1)
            qk = dist1.pdf(domain)

        # Map extremely small values to non-zero to avoid inf quotients
        pk = list(map(lambda k: 1e-300 if k < 1e-300 else k, pk))
        qk = list(map(lambda k: 1e-300 if k < 1e-300 else k, qk))

        kl0 = entropy(qk, pk)
        kl1 = entropy(pk, qk)

        kl_arr[i] = np.mean([kl0, kl1])
        c_count += 1

    print("Computing KLD for discrete columns...")
    for i in d_cols:
        # Convert columns to frequency tables
        dist0 = Counter(map(lambda r: r[i], X_0))
        dist1 = Counter(map(lambda r: r[i], X_1))
        
        # Get domain
        domain = set(dist0) | set(dist1)

        # Map extremely small values to non-zero to avoid inf quotients
        for x in domain:
            dist0[x] = 1e-300 if dist0[x] < 1e-300 else dist0[x]
            dist1[x] = 1e-300 if dist1[x] < 1e-300 else dist1[x]
        pk = list(sorted(map(lambda c: dist0[c] / sum(dist0.values()),\
            domain)))
        qk = list(sorted(map(lambda c: dist1[c] / sum(dist1.values()),\
            domain)))

        kl0 = entropy(qk, pk)
        kl1 = entropy(pk, qk)

        kl_arr[i] = np.mean([kl0, kl1])

    return kl_arr


if __name__ == '__main__':
    main()
