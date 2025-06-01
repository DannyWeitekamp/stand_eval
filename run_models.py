import numpy as np
from sklearn.model_selection import train_test_split
from gen_synth_conds import gen_synthetic_dnf_data, print_dnf
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# from stand.tree_classifier import TreeClassifier
from stand.stand import STANDClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# -----------------------------------------------------------
# Certainty Stats for Thrashing, TPR etc. 
def eval_total_cert_stats(corrs, holdout_certs, cert_threshs=[.9,1.0]):
    # corrs = np.array([rew > 0 for ind, rew in skill_app_map.values()], dtype=np.bool_)
    incorrs = ~corrs
    L = len(holdout_certs)
    
    # Vacillation Rate : Proportion of errors that reoccur
    FP_reocc    = np.zeros(L-1, dtype=np.float64)
    FN_reocc    = np.zeros(L-1, dtype=np.float64)
    error_reocc = np.zeros(L-1, dtype=np.float64)
    total_FP_reocc, total_pTPs = 0, 0 
    total_FN_reocc, total_pTNs = 0, 0 
    total_error_reocc, total_pTs = 0, 0 

    # Productive Monotonicity: Proportion of certainty changes
    #  that correctly move toward 1.0 if correct or -1.0 if incorrect. 
    prod_monot  = np.zeros(L-1, dtype=np.float64)
    total_prod_d, total_d  = 0, 0
    
    # Certainty Preds @ thresh 
    precisions_at_thresh  = [np.zeros(L, dtype=np.float64) for _ in range(len(cert_threshs))]
    total_TP_at_thresh = [0] * len(cert_threshs)
    total_PP_at_thresh = [0] * len(cert_threshs)

    for i, certs in enumerate(holdout_certs):
        # Fill in missing certs with 0 predictions
        certs = np.pad(certs, (0, len(corrs)-len(certs)), constant_values=(0, 0))
        TPs = certs[corrs] > 0.0
        TNs = certs[incorrs] < 0.0

        # Calc delta based measures
        if(i != 0):
            k = i-1

            # -- Error Re-occurance ---
            FP_reoccs = ~TNs[prev_TNs] # Neg that were ok then pos
            FN_reoccs = ~TPs[prev_TPs] # Pos that were ok then neg
            n_FP_reoccs, n_pTNs = np.count_nonzero(FP_reoccs), len(prev_TNs)
            n_FN_reoccs, n_pTPs = np.count_nonzero(FN_reoccs), len(prev_TPs)
            n_reoccs, n_pTs = n_FP_reoccs + n_FN_reoccs, n_pTPs + n_pTNs

            FP_reocc[k] = (n_FP_reoccs / n_pTNs) if n_pTNs > 0 else 0.0
            FN_reocc[k] = (n_FN_reoccs / n_pTPs) if n_pTPs > 0 else 0.0
            error_reocc[k] = (n_reoccs / n_pTs) if n_pTs > 0 else 0.0

            total_FP_reocc += n_FP_reoccs
            total_pTPs += n_pTPs
            total_FN_reocc += n_FN_reoccs
            total_pTNs += n_pTNs
            total_error_reocc += n_reoccs
            total_pTs += n_pTs

            # --- Productive Monotonicity ---             
            d_certs = certs-prev_certs
            print(certs)
            prod_pos = (d_certs > 0) & corrs
            prod_neg = (d_certs < 0) & incorrs
            prods = (prod_pos | prod_neg)[d_certs != 0]
            n_prod_d, n_d = np.count_nonzero(prods), len(prods)

            prod_monot[k] = (n_prod_d / n_d) if n_d > 0 else 1.0
            total_prod_d += n_prod_d
            total_d += n_d

        # Update Precision
        for t, thresh in enumerate(cert_threshs):
            pred = certs >= thresh;

            # True positives over predicted positives
            TP = np.count_nonzero(pred & corrs)
            PP = np.count_nonzero(pred)
            total_TP_at_thresh[t] += TP
            total_PP_at_thresh[t] += PP
            precisions_at_thresh[t][i] = (TP / PP) if PP > 0 else 1.0

        prev_certs = certs
        prev_TPs = TPs
        prev_TNs = TNs

    out = {}
    for t, thresh in enumerate(cert_threshs):
        total_TP = total_TP_at_thresh[t]
        total_PP = total_PP_at_thresh[t]

        out[("total_precision", thresh)] = (total_TP / total_PP) if total_PP > 0 else 1.0
        out[("precision", thresh)] = precisions_at_thresh[t]

    out.update({
        "FP_reocc" : FP_reocc,
        "FN_reocc" : FN_reocc,
        "error_reocc" : error_reocc,
        "total_FP_reocc" : total_FP_reocc / total_pTNs if total_pTPs > 0 else 1.0,
        "total_FN_reocc" : total_FN_reocc / total_pTPs if total_pTPs > 0 else 1.0,
        "total_error_reocc" : total_error_reocc / total_pTs if total_pTs > 0 else 1.0,

        "prod_monot" : prod_monot,
        "total_prod_monot" : (total_prod_d / total_d) if total_d > 0 else 1.0,
    })

    return out


one_hot_encoder = OneHotEncoder()


X, y, dnf = gen_synthetic_dnf_data(
                          n_samples=2000,
                          n_feats=1000,
                          vals_per_feat=3,
                          conj_len=4, 
                          num_conj=3,
                          conj_probs=.15,
                          dupl_lit_prob=0.2,
                          force_same_vals=False)

one_hot_encoder.fit(X)
# X_one_hot = one_hot_encoder.transform(X).toarray()

# print("X")
# print(X)



data = train_test_split(X, y, train_size=100)

def rf_cert_fn(classifier, X_nom_subset):
    # print("RF")
    probs = classifier.predict_proba(X_nom_subset)
    labels = classifier.classes_
    best_ind = np.argmax(probs, axis=1)
    p = probs[:, best_ind]
    return np.where(best_ind == 0, -p, p)

def xg_cert_fn(classifier, X_nom_subset):
    # print("XG")
    probs = classifier.predict_proba(X_nom_subset)
    labels = classifier.classes_
    # labels  = self.le.inverse_transform(labels)
    best_ind = np.argmax(probs, axis=1)
    p = probs[:, best_ind]
    return np.where(best_ind == 0, -p, p)

def stand_cert_fn(classifier, X_nom_subset):
    # print("STAND")
    probs, labels  = classifier.predict_prob(X_nom_subset, None)

    # print(probs.shape, labels)
    # probs = probs[0]
    best_ind = np.argmax(probs, axis=1)
    # print(best_ind)
    p = probs[:, best_ind]
    return np.where(best_ind == 0, -p, p)
    # return labels[best_ind] * probs[:, best_ind]


models = {
    "stand" : {"model": STANDClassifier(), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
    # "xgboost" : {"model": XGBClassifier(), "is_stand" : False, "one_hot" : True, "cert_fn" : xg_cert_fn},
    # "rand_forest" : {"model": RandomForestClassifier(), "is_stand" : False, "one_hot" : True, "cert_fn" : rf_cert_fn},
    # "decision_tree" : {"model": DecisionTreeClassifier(), "is_stand" : False, "one_hot" : True, "cert_fn" : None },
}


def test_model(name, config, data, is_stand=False):
    X_train, X_test, y_train, y_test = data

    model = config['model']
    is_stand = config['is_stand']
    one_hot = config['one_hot']
    cert_fn = config['cert_fn']
    holdout_certs = []
    holdout_accuracies = []

    # One-hot encode models that aren't stand (maybe should get rid of this
    #   the issue is that some of the models don't quite work without one-hot)
    if(one_hot):
        X_train = one_hot_encoder.transform(X_train).toarray()
        X_test = one_hot_encoder.transform(X_test).toarray()
    
    for i in range(1,len(X_train)+1):
        # Fit on subset of training set 
        X_train_slc = X_train[0:i]
        y_train_slc = y_train[0:i]
        if(is_stand):
            model.fit(X_train_slc.astype(dtype=np.int32), None, y_train_slc)
            # print(model)
        else:
            model.fit(X_train_slc, y_train_slc)

        # Test on the test_set
        if(is_stand):
            y_preds = model.predict(X_test, None)
        else:
            y_preds = model.predict(X_test)

        corrs = y_preds == y_test

        if(cert_fn):
            y_certs = cert_fn(model, X_test)

            # print("y_certs", y_certs)
            holdout_certs.append(y_certs)
        holdout_accuracies.append(accuracy_score(y_test, y_preds))



    # print("FP_reocc:", stats["FP_reocc"])
    # print("FN_reocc:", stats["FN_reocc"])
    # print("error_reocc:", stats["error_reocc"])
        

    # "prod_monot" : prod_monot,
    # "total_prod_monot" : (total_prod_d / total_d) if total_d > 0 else 1.0,

    # print(stats)

    if(is_stand):
        print(model)
    print(name)
    print("Accuracies: ", holdout_accuracies)
    print("Accuracy: ", holdout_accuracies[-1])
    if(cert_fn is not None):
        stats = eval_total_cert_stats(y_test==1.0, holdout_certs)
        print("total_FP_reocc:", stats["total_FP_reocc"])
        print("total_FN_reocc:", stats["total_FN_reocc"])
        print("total_error_reocc:", stats["total_error_reocc"])
        print("prod_monot:", stats["prod_monot"])
        print("total_prod_monot:", stats["total_prod_monot"])

print_dnf(dnf)


X_train, X_test, y_train, y_test = data


from numba import njit
import numpy as np

@njit(cache=True)
def index_of(array, item):
    for idx, val in enumerate(array):
        if val == item:
            return idx

def ensure_first_neg_pos(X_train, y_train):
    first_0 = index_of(y_train, 0)
    print("first_0", first_0)
    X_train[[0, first_0]] = X_train[[first_0, 0]]
    y_train[[0, first_0]] = y_train[[first_0, 0]]

    first_1 = index_of(y_train, 1)

    # Swap
    
    X_train[[1, first_1]] = X_train[[first_1, 1]]
    y_train[[1, first_1]] = y_train[[first_1, 1]]


ensure_first_neg_pos(X_train, y_train)



print("y_train", y_train)

# raise ValueError()

print("=1 Prop", np.sum(y_train)/len(y_train), np.sum(y_test)/len(y_test))

for name, model in models.items():
    is_stand = name == "stand"
    test_model(name, model, data, is_stand)


# print(X)
# print(y)

