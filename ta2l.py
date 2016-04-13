from __future__ import division

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import chi2
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM


def main():
    X_train, y_train, X_test, id_test = init()
    #X_train, X_test = preprocess(X_train, y_train, X_test)
    split_data = split_data_types(X_train, X_test)
    X_train_cont, X_train_disc, X_test_cont, X_test_disc = split_data

    from sklearn.cross_validation import train_test_split
    X_train1, X_train2, y_train1, y_train2 = train_test_split(X_train,\
        y_train, test_size=0.4)

    # classifier
    clf = xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=350,\
        learning_rate=0.03, nthread=4, subsample=0.95, colsample_bytree=0.85,\
        seed=4242)

    X_fit, X_eval, y_fit, y_eval= train_test_split(X_train1, y_train1,\
        test_size=0.3)

    # fitting
    print("== Training Classifier ==")
    clf.fit(X_train1, y_train1, early_stopping_rounds=20, eval_metric="auc",\
        eval_set=[(X_eval, y_eval)])

    auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:,1])
    print('Overall AUC:', auc)

    y_pred = clf.predict_proba(X_test)[:,1]
    submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred})
    submission.to_csv("submission.csv", index=False)

    print('Completed!')

def init():
    archfile = 'archive.pckl'
    if os.path.isfile(archfile):
        print("== Reloading Data ==")
        archive = pickle.load(open(archfile, 'rb'))
    else:
        print("== Loading Data ==")
        archive = load_data()
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
    
    cols = df_train.columns
    for i in range(len(cols)):
        if cols[i] == 'var15':
            print(i)
            break

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

    # Create new ndarray for each case
    X_train_cont = np.ndarray(shape=(X_train.shape[0], sum(col_is_cont)),\
            dtype=np.float64)
    X_train_disc = np.ndarray(shape=(X_train.shape[0], len(col_is_cont) -\
            sum(col_is_cont)), dtype=np.float64)

    X_test_cont = np.ndarray(shape=(X_train.shape[0], sum(col_is_cont)),\
            dtype=np.float64)
    X_test_disc = np.ndarray(shape=(X_train.shape[0], len(col_is_cont) -\
            sum(col_is_cont)), dtype=np.float64)

    # Copy columns appropriately
    c_idx = 0
    d_idx = 0
    for c in range(len(col_is_cont)):
        if col_is_cont[c]:
            X_train_cont[:,c_idx] = X_train[:,c]
            X_test_cont[:,c_idx] = X_train[:,c]
            c_idx += 1
        else:
            X_train_disc[:,d_idx] = X_train[:,c]
            X_test_disc[:,d_idx] = X_train[:,c]
            d_idx += 1

    return X_train_cont, X_train_disc, X_test_cont, X_test_disc

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


if __name__ == '__main__':
    main()
