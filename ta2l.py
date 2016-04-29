import operator
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
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import chi2
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from sklearn.ensemble import VotingClassifier
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin


tst = 0

class VotingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers, weights=None):
        self.clfs = classifiers
        self.weights = weights

    def fit(self, arg_arr):
        pairs = zip(self.clfs, arg_arr)
        for clf, args in pairs:
            clf.fit(**args)

    def predict(self, X):
        self.classes_ = np.asarray([clf.predict(X) for clf in self.clfs])
        if self.weights:
            avg = self.predict_proba(X)

            majority = np.apply_along_axis(lambda x: max(enumerate(x),\
                key=operator.itemgetter(1))[0], axis=1, arr=avg)

        else:
            majority = np.asarray([np.argmax(np.bincount(self.classes_[:,c]))\
                for c in range(self.classes_.shape[1])])

        return majority

    def predict_proba(self, X):
        def vote(t):
            global tst
            from collections import defaultdict
            count = Counter()
            for i in range(len(t)):
                val = 0 if t[i][0] > t[i][1] else 1
                count[val] += self.weights[i]
            vote = count.most_common(1)[0][0]
            if vote:
                global tst
                tst += 1
                out = max(t, key=lambda x: x[1])
            else:
                out = min(t, key=lambda x: x[1])

            return out

        probs = [clf.predict_proba(X) for clf in self.clfs]
        self.probas_ = zip(*probs)
        majority = np.asarray([vote(x) for x in self.probas_])

        return majority


def main():
    #X_train, y_train, X_test, id_test = init()

    kl_archive = kl_init()
    X_train = kl_archive[0]
    y_train = kl_archive[1]
    X_test = kl_archive[2]
    id_test = kl_archive[3]
    c_cols = kl_archive[4]
    d_cols = kl_archive[5]
    kl_arr = kl_archive[6]

    X_train, X_test, kl_map = kl_filter(X_train, X_test, kl_arr, n=(0, 80))
    #X_train, X_test = preprocess(X_train, y_train, X_test)

    X_fit, X_valid, y_fit, y_valid = train_test_split(X_train,\
        y_train, test_size=0.3)
    X_fit, X_eval, y_fit, y_eval= train_test_split(X_fit, y_fit,\
        test_size=0.3)

    # classifiers
    xgb_clf = xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=1000,\
        learning_rate=0.03, nthread=4, subsample=0.95, colsample_bytree=0.85,\
        silent=True)
    lr_clf = LogisticRegression(penalty='l1', class_weight='balanced',\
        random_state=1, n_jobs=4)
    rf_clf = RandomForestClassifier(random_state=1, n_jobs=4)
    bnb_clf = BernoulliNB(alpha=10) 

    eclf = VotingClassifier(classifiers=[xgb_clf, lr_clf, rf_clf, bnb_clf],\
        weights=[1,1,2,1])
    #eclf = VotingClassifier(classifiers=[xgb_clf], weights=[1])

    # fitting
    print("== Training Classifier ==")
    '''
    fit_args = [
            {
                'X': X_fit,
                'y': y_fit,
                'early_stopping_rounds': 100,
                'eval_metric': "auc",
                'eval_set': [(X_eval, y_eval)]
            },
            {'X': X_fit, 'y': y_fit},
            {'X': X_fit, 'y': y_fit},
            {'X': X_fit, 'y': y_fit}
    ]

    eclf.fit(fit_args)

    global tst
    tst = 0
    auc = roc_auc_score(y_valid, eclf.predict_proba(X_valid)[:,1])
    print("Validation Class 1 count:", tst, "out of", X_valid.shape[0])
    print('Validation AUC:', auc)
    '''
    
    fit_args = [
            {
                'X': X_train,
                'y': y_train,
                'early_stopping_rounds': 100,
                'eval_metric': "auc",
                'eval_set': [(X_eval, y_eval)]
            },
            {'X': X_train, 'y': y_train},
            {'X': X_train, 'y': y_train},
            {'X': X_train, 'y': y_train}
    ]

    eclf.fit(fit_args)

    tst = 0
    y_pred = eclf.predict_proba(X_test)[:,1]
    print("Testing Class 1 count:", tst, "out of", len(y_pred))
    submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred})
    submission.to_csv("submission.csv", index=False)

    print('Completed!')


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

def kl_filter(X_train, X_test, kl_arr, n=50):
    if type(n) == int:
        l = 0
        r = n
    elif type(n) == tuple or type(n) == list:
        l = min(n)
        r = max(n)

    print("== Reducing Features ==")
    kl_map = list(sorted(zip(kl_arr, range(len(kl_arr))), reverse=True))
    X_train_fil = np.ndarray(shape=(X_train.shape[0], r-l), dtype=np.float64)
    X_test_fil = np.ndarray(shape=(X_test.shape[0], r-l), dtype=np.float64)

    f_idx = 0
    for i in range(l, r):
        idx = kl_map[i][1]
        X_train_fil[:,f_idx] = X_train[:,idx]
        X_test_fil[:,f_idx] = X_test[:,idx]
        f_idx += 1

    return X_train_fil, X_test_fil, kl_map

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

        # Get 1000 evenly spaced values from the observed domain
        left = min(col_0[0], col_1[0])
        right = max(col_0[-1], col_1[-1])
        domain = np.linspace(left, right, 1000)

        # Estimate feature's distributions using Gaussian Kernel Density
        # Estimation. If a feature is constant for a class, use a so-called
        # "finite delta function" for its distribution. (Pr(x) = 1 for x == c
        # and Pr(x) = 0 for x != c)
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
