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

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.svm import OneClassSVM


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import normalize

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold

from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score


def main():
    #X_train, y_train, X_test, id_test = init()

    #kl_archive = kl_init()
    df_train, df_test = init()

    #df_train, df_test = preprocess(df_train, df_test)

    '''
    X_train = df_to_ndarray(df_train, df_train.columns[1:-1])
    y_train = df_to_ndarray(df_train, df_train.columns[-1])
    X_test = df_to_ndarray(df_test, df_test.columns[1:])
    y_test = df_to_ndarray(df_test, df_test.columns[0])
    #c_cols = kl_archive[4]
    #d_cols = kl_archive[5]
    #kl_map = kl_archive[6]

    #X_train, X_test, kl_map = kl_filter(X_train, X_test, kl_map, n=(0, 100))

    X_fit, X_valid, y_fit, y_valid = train_test_split(X_train,\
        y_train, test_size=0.3)
    X_fit, X_eval, y_fit, y_eval= train_test_split(X_fit, y_fit,\
        test_size=0.3)

    # classifiers
    xgb_clf = xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=350,\
        learning_rate=0.025, nthread=4, subsample=0.9, colsample_bytree=0.85,\
        silent=True)
    lr_clf = LogisticRegression(penalty='l1', class_weight='balanced',\
        random_state=1, n_jobs=4)
    rf_clf = RandomForestClassifier(random_state=1, n_jobs=4)
    #bnb_clf = BernoulliNB(alpha=0.1) 
    mnb_clf = MultinomialNB(alpha=1)
    knn_clf = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')

    #eclf = VotingClassifier(classifiers=[xgb_clf, lr_clf, rf_clf, mnb_clf],\
    #    weights=[2,1,1,1])
    eclf = VotingClassifier(classifiers=[xgb_clf], weights=[1])
    eclf = xgb_clf

    # fitting

    print("== Training Classifier ==")
    fit_args = [
            {
                'X': X_train,
                'y': y_train,
                'early_stopping_rounds': 50,
                'eval_metric': "auc",
                'eval_set': [(X_eval, y_eval)],
                'verbose': False
            }
            #},
            #{'X': X_fit, 'y': y_fit},
            #{'X': X_fit, 'y': y_fit},
            #{'X': X_fit, 'y': y_fit}
    ]

    #eclf.fit(fit_args)
    eclf.fit(**(fit_args[0]))

    print("== Validating ==")
    auc = roc_auc_score(y_valid, eclf.predict_proba(X_valid)[:,1])
    print('Validation AUC:', auc)
    '''

    '''
    fit_args = [
            {
                'X': X_train,
                'y': y_train,
                'early_stopping_rounds': 50,
                'eval_metric': "auc",
                'eval_set': [(X_eval, y_eval)],
                'verbose': False
            }
            #},
            #{'X': X_train, 'y': y_train},
            #{'X': X_train, 'y': y_train},
            #{'X': X_train, 'y': y_train}
    ]

    #eclf.fit(fit_args)
    eclf.fit(**(fit_args[0]))
    '''

    #eclf = RandomForestClassifier(random_state=1, n_jobs=4)
    eclf = xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=350,\
        learning_rate=0.025, nthread=4, subsample=0.9, colsample_bytree=0.85,\
        silent=True)
    y_pred = run_classifier(eclf, df_train, df_test)
    id_test = df_to_ndarray(df_test, df_test.columns[0])

    #y_pred = eclf.predict_proba(X_test)[:,1]
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
        kl_map = kl_diverge(X_train, y_train, c_cols, d_cols)
        kl_archive = archive + [c_cols, d_cols, kl_map]
        pickle.dump(kl_archive, open(archfile, 'wb'))

    return kl_archive

def init():
    archfile = 'archive.pckl'
    if os.path.isfile(archfile):
        print("== Reloading Data ==")
        archive = pickle.load(open(archfile, 'rb'))
    else:
        print("== Loading Data ==")
        df_train, df_test = load_data()
        archive = [
            df_train,
            df_test
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

    return df_train, df_test

def get_cont_disc_cols(df_train, df_test):
    def get_cont(df, cols):
        c_cols = set()
        for c in cols:
            for x in df[c].values:
                if int(x) != x:
                    c_cols.add(c)
                    break
        return c_cols


    print("Splitting continuous and discrete...")

    # Find continuous columns
    train_cols = set(df_train.columns)
    test_cols = set(df_train.columns)
    cols = train_cols & test_cols
    cols -= set(['TARGET', 'ID'])

    c_cols = get_cont(df_train, cols)
    c_cols |= get_cont(df_test, cols)

    d_cols = cols - c_cols

    return c_cols, d_cols

def preprocess(df_train, df_test):
    print("== Preprocessing Data ==")

    c_cols, d_cols = map(list, get_cont_disc_cols(df_train, df_test))

    print("Running PCA...")
    X_cont = pd.concat((df_train[c_cols], df_test[c_cols]), axis=0)
    sep = df_train.shape[0]

    pca = PCA(n_components=5)
    X_proj = pca.fit_transform(normalize(X_cont, axis=0))
    for i in range(X_proj.shape[1]):
        df_train.insert(1, 'PCA%d'%i, X_proj[:sep,i])
        df_test.insert(1, 'PCA%d'%i, X_proj[sep:,i])

    '''
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
    '''

    '''
    from sklearn.decomposition import IncrementalPCA
    ipca = IncrementalPCA(n_components=3)
    ipca.fit(X_merge)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]+3))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]+3))
    
    X_train[:,-3:] = ipca.transform(X_train)
    X_test[:,-3:] = ipca.transform(X_test)
    '''

    '''
    nmzr = Normalizer(norm='l2')
    X_merge = nmzr.fit_transform(X_merge)
    
    scale = MinMaxScaler()
    X_merge = scale.fit_transform(X_merge)

    X_train = X_merge[:sep]
    X_test = X_merge[sep:]
    
    return X_train, X_test
    '''
    
    return df_train, df_test

def kl_diverge(df_train, c_cols, d_cols):
    rows_1 = filter(lambda i: df_train['TARGET'][i], range(df_train.shape[0]))
    rows_1 = set(rows_1)
    rows_0 = set(range(df_train.shape[0])) - rows_1

    kl_map = {}

    print("Computing KLD for continuous columns...")
    c_count = 0
    for c in c_cols:
        print("%d of %d" % (c_count, len(c_cols)))
        # Get column vector for each class
        col_0 = df_train[c][rows_0]
        col_1 = df_train[c][rows_1]

        # Get domain to compute probabilities on
        left = min(col_0.min(), col_1.min())
        right = max(col_0.max(), col_1.max())
        domain = np.linspace(left, right, 1000)
    
        # Estimate feature's distributions using Gaussian Kernel Density
        # Estimation. If a feature is constant for a class, use a so-called
        # "finite delta function" for its distribution. (Pr(x) = 1 for x == c
        # and Pr(x) = 0 for x != c)
        if col_0.std() == 0:
            pk = list(map(lambda x: 1 if x in elem_0 else 0, domain))
        else:
            dist_0 = gaussian_kde(col_0)
            pk = dist_0.pdf(domain)
        if col_1.std() == 0:
            qk = list(map(lambda x: 1 if x in elem_1 else 0, domain))
        else:
            dist_1 = gaussian_kde(col_1)
            qk = dist_1.pdf(domain)

        # Map extremely small values to non-zero to avoid inf quotients
        pk = list(map(lambda k: 1e-300 if k < 1e-300 else k, pk))
        qk = list(map(lambda k: 1e-300 if k < 1e-300 else k, qk))

        kl_0 = entropy(qk, pk)
        kl_1 = entropy(pk, qk)

        kl_map[c] = np.mean([kl_0, kl_1])
        c_count += 1

    print("Computing KLD for discrete columns...")
    for c in d_cols:
        # Convert columns to frequency tables
        dist_0 = Counter(df_train[c][rows_0])
        dist_1 = Counter(df_train[c][rows_1])

        # Get domain
        domain = set(dist_0) | set(dist_1)

        # Map extremely small values to non-zero to avoid inf quotients
        for x in domain:
            dist_0[x] = 1e-300 if dist_0[x] < 1e-300 else dist_0[x]
            dist_1[x] = 1e-300 if dist_1[x] < 1e-300 else dist_1[x]
        pk = list(map(lambda c: dist_0[c] / sum(dist_0.values()),\
            sorted(domain)))
        qk = list(map(lambda c: dist_1[c] / sum(dist_1.values()),\
            sorted(domain)))

        kl_0 = entropy(qk, pk)
        kl_1 = entropy(pk, qk)

        kl_map[c] = np.mean([kl_0, kl_1])

    return kl_map

def kl_filter(X_train, X_test, kl_map, n=50):
    if type(n) == int:
        l = 0
        r = n
    elif type(n) == tuple or type(n) == list:
        l = min(n)
        r = max(n)

    print("== Reducing Features ==")
    kl_map = list(sorted(zip(kl_map, range(len(kl_map))), reverse=True))
    X_train_fil = np.ndarray(shape=(X_train.shape[0], r-l), dtype=np.float64)
    X_test_fil = np.ndarray(shape=(X_test.shape[0], r-l), dtype=np.float64)

    f_idx = 0
    for i in range(l, r):
        idx = kl_map[i][1]
        X_train_fil[:,f_idx] = X_train[:,idx]
        X_test_fil[:,f_idx] = X_test[:,idx]
        f_idx += 1

    return X_train_fil, X_test_fil, kl_map

def df_to_ndarray(df, cols):
    return df[cols].values

def run_classifier(clf, df_train, df_test):
    mk_xtrain = lambda df: df_to_ndarray(df, df.columns[1:-1])
    mk_ytrain = lambda df: df_to_ndarray(df, df.columns[-1])
    mk_xtest = lambda df: df_to_ndarray(df, df.columns[1:])

    X_train = mk_xtrain(df_train)
    y_train = mk_ytrain(df_train)
    X_test = mk_xtest(df_test)

    train_preds = None
    test_preds = None

    cycle = 0
    skf = StratifiedKFold(y_train, n_folds=10)
    for trn_rows, tst_rows in skf:
        print("Cycle", cycle)
        ilc_vsbl = df_train.iloc(trn_rows)

        vsbl_train = ilc_vsbl[:][df_train.columns[1:-1]].values
        vsbl_tgt = ilc_vsbl[:]['TARGET'].values

        ilc_blnd = df_train.iloc(tst_rows)
        blnd_train = ilc_blnd[:][df_train.columns[1:-1]].values
        blnd_tgt = ilc_blnd[:]['TARGET'].values

        clf.fit(vsbl_train, vsbl_tgt, early_stopping_rounds=50,\
            eval_metric='auc', eval_set=[(blnd_train, blnd_tgt)],\
            verbose=False)
        #clf.fit(vsbl_train, vsbl_tgt)
        blnd_pred = clf.predict_proba(blnd_train)[:,1]

        print('Blind Log Loss:', log_loss(blnd_tgt, blnd_pred))
        print('Blind AUC:', roc_auc_score(blnd_tgt, blnd_pred))

        if type(train_preds) == type(None) and type(test_preds) == type(None):
            train_preds = clf.predict_proba(X_train)[:,1]
            test_preds = clf.predict_proba(X_test)[:,1]
        else:
            train_preds *= clf.predict_proba(X_train)[:,1]
            test_preds *= clf.predict_proba(X_test)[:,1]
        cycle += 1

    train_preds = np.power(train_preds, 1./cycle)
    test_preds = np.power(test_preds, 1./cycle)

    print('Average Log Loss:', log_loss(df_train.TARGET.values, train_preds))
    print('Average AUC:', roc_auc_score(df_train.TARGET.values, train_preds))

    return test_preds


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

            count = Counter()
            for i in range(len(t)):
                val = 0 if t[i][0] > t[i][1] else 1
                count[val] += self.weights[i]
            vote = count.most_common(1)[0][0]
            print(t, count, vote)
            if vote:
                out = max(t, key=lambda x: x[1])
            else:
                out = min(t, key=lambda x: x[1])

            return out

        probs = [clf.predict_proba(X) for clf in self.clfs]
        self.probas_ = zip(*probs)
        majority = np.asarray([vote(x) for x in self.probas_])

        return majority


if __name__ == '__main__':
    main()
