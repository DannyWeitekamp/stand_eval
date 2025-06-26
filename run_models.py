import numpy as np
from numpy.random import poisson
from sklearn.model_selection import train_test_split
from gen_synth_conds import gen_synthetic_dnf_data, print_dnf
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# from stand.tree_classifier import TreeClassifier
from stand.stand import STANDClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle
from datetime import datetime
import os
from random import random as py_random
from numba import njit

import time
class PrintElapse():
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.t0 = time.time_ns()/float(1e6)
    def __exit__(self,*args):
        self.t1 = time.time_ns()/float(1e6)
        print(f'{self.name}: {self.t1-self.t0:.6f} ms')

def front_load_neg(X, y, neg_bias=.8):
    # new_inds = []
    pos_inds = np.nonzero(y==1)[0]
    neg_inds = np.nonzero(y!=1)[0]
    # for i in range(len(y)):

    inds = np.concatenate([pos_inds, neg_inds])
    weights = np.concatenate([
        (1.0-neg_bias)*np.ones(len(pos_inds)),
        neg_bias*np.ones(len(neg_inds))
    ])
    weights /= np.sum(weights)
    # print(len(inds), len(y))
    new_inds = np.random.choice(inds, size=len(y), replace=False, p=weights)
    # print("BEFORE:")
    # print(y)
    # print("AFTER:")
    # print(y[new_inds])
    # raise ValueError()
    return X[new_inds], y[new_inds]

        





def biased_train_test_split(X, y, train_size=100, 
                # true_prop=.5 #.8
                true_prop=.8,
                ):
    n_true = int(train_size*(true_prop))
    # print(np.nonzero(y==1)[0])
    pos_inds = np.random.choice(np.nonzero(y==1)[0], size=n_true, replace=False)
    # print(pos_inds)
    n_false = int(train_size-n_true)
    neg_inds = np.random.choice(np.nonzero(y!=1)[0], size=n_false, replace=False)
    # print(neg_inds)

    train_inds = np.concatenate([pos_inds, neg_inds])
    np.random.shuffle(train_inds)
    # print("train_inds:", train_inds)
    test_mask = np.ones(len(X), dtype=np.bool_)
    test_mask[train_inds] = 0

    # if(front_load_neg):
    #     new_train_inds = []
    #     y_train = y[train_inds]
    #     pos_inds = y_train[]



    # print(y[train_inds])

    # print(X[train_inds].shape, y[train_inds].shape, X[test_mask].shape, y[test_mask].shape)
    return X[train_inds], X[test_mask], y[train_inds], y[test_mask]





DEFAULT_CERT_BINS = [
    # (.50, .60), (.60, .70), (.70, .80), (.80, .90),
    (.50, .55), (.55, .60), (.60, .65), (.65, .70), (.70, .75), (.75, .80), (.80, .85), (.85, .90),
    (.90, .92), (.92, .94), (.94, .96), (.96, .98), (.98, 1.0), (1.0, 1.0)]

# -----------------------------------------------------------
# Certainty Stats for Thrashing, TPR etc. 
def eval_total_cert_stats(corrs, holdout_certs, cert_bins=DEFAULT_CERT_BINS, diff_thresh=0.05):
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
    w_prod_monot  = np.zeros(L-1, dtype=np.float64)
    total_prod_d, total_w_prod_d, total_d, total_abs_d_cert  = 0, 0.0, 0, 0.0
    
    # Certainty Preds @ thresh 
    precisions_in_bin  = [np.zeros(L, dtype=np.float64) for _ in range(len(cert_bins))]
    total_TP_at_thresh = [0] * len(cert_bins)
    total_PP_at_thresh = [0] * len(cert_bins)

    for i, certs in enumerate(holdout_certs):
        # Fill in missing certs with 0 predictions
        # certs = np.pad(certs, (0, len(corrs)-len(certs)), constant_values=(0, 0))
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

            # print("::", certs.shape, prev_certs.shape)

            # print(certs)
            prod_pos = (d_certs > 0) & corrs
            prod_neg = (d_certs < 0) & incorrs
            prod_mask = (prod_pos | prod_neg)
            prods = prod_mask[np.abs(d_certs) > diff_thresh]
            n_prod_d, n_d = np.count_nonzero(prods), len(prods)            
            prod_monot[k] = (n_prod_d / n_d) if n_d > 0 else np.nan


            abs_d_certs = np.abs(d_certs)
            sum_abs_d_certs = np.sum(abs_d_certs)
            w_prod_d = (np.sum(prod_mask*abs_d_certs) - np.sum((~prod_mask)*abs_d_certs))
            w_prod_monot[k] = w_prod_d / sum_abs_d_certs if sum_abs_d_certs != 0.0 else np.nan
            total_w_prod_d += w_prod_d #/// len(d_certs)
            total_abs_d_cert += sum_abs_d_certs
            # print("::", w_prod_d, sum_abs_d_certs)


            # print(prod_monot[k], np.mean(np.abs((prod_pos | prod_neg)*d_certs))-np.mean(np.abs((~(prod_pos | prod_neg))*d_certs)), n_d)
            total_prod_d += n_prod_d
            total_d += n_d

        # Update Precision
        for t, c_bin in enumerate(cert_bins):
            c_min, c_max = c_bin
            pred = (certs >= c_min) & (certs < c_max);

            # True positives over predicted positives
            TP = np.count_nonzero(pred & corrs)
            PP = np.count_nonzero(pred)
            total_TP_at_thresh[t] += TP
            total_PP_at_thresh[t] += PP
            precisions_in_bin[t][i] = (TP / PP) if PP > 0 else 1.0

        prev_certs = certs
        prev_TPs = TPs
        prev_TNs = TNs

    out = {}
    for t, c_bin in enumerate(cert_bins):
        total_TP = total_TP_at_thresh[t]
        total_PP = total_PP_at_thresh[t]

        out[("total_precision", c_bin)] = (total_TP / total_PP) if total_PP > 0 else 1.0
        out[("precision", c_bin)] = precisions_in_bin[t]
        out[("bin_n", c_bin)] = total_PP
        out[("TP_n", c_bin)] = total_TP

    out.update({
        "FP_reocc" : FP_reocc,
        "FN_reocc" : FN_reocc,
        "error_reocc" : error_reocc,
        "total_FP_reocc" : total_FP_reocc / total_pTNs if total_pTPs > 0 else 1.0,
        "total_FN_reocc" : total_FN_reocc / total_pTPs if total_pTPs > 0 else 1.0,
        "total_error_reocc" : total_error_reocc / total_pTs if total_pTs > 0 else 1.0,
        "prod_monot" : prod_monot,
        "w_prod_monot" : w_prod_monot,
        "total_prod_monot" : (total_prod_d / total_d) if total_d > 0 else 1.0,
        "total_w_prod_monot" : (total_w_prod_d / total_abs_d_cert) if total_d > 0 else 1.0,
    })

    return out




def rf_cert_fn(classifier, X_nom_subset):
    # print("RF")
    probs = classifier.predict_proba(X_nom_subset)
    labels = classifier.classes_
    best_ind = np.argmax(probs, axis=1)
    p = probs[np.arange(len(probs)), best_ind]

    return np.where(best_ind == 0, -p, p)

def xg_cert_fn(classifier, X_nom_subset):
    # print("XG")
    probs = classifier.predict_proba(X_nom_subset)
    labels = classifier.classes_
    # labels  = self.le.inverse_transform(labels)
    best_ind = np.argmax(probs, axis=1)
    p = probs[np.arange(len(probs)), best_ind]
    return np.where(best_ind == 0, -p, p)

def stand_cert_fn(classifier, X_nom_subset):
    # print("STAND")
    probs, labels  = classifier.predict_proba(X_nom_subset, None)

    # print(probs.shape, labels)
    # probs = probs[0]
    best_ind = np.argmax(probs, axis=1)
    # print(best_ind[:20])

    # p = np.take(probs, best_ind, axis=1)
    p = probs[np.arange(len(probs)), best_ind]
    # p = np.take_along_axis(probs, best_ind, axis=1)


    # print(probs.shape, p.shape, best_ind.shape)
    return np.where(best_ind == 0, -p, p)
    # return labels[best_ind] * probs[:, best_ind]

def dt_cert_fn(classifier, X_nom_subset):
    preds = classifier.predict(X_nom_subset)
    return np.where(preds==1, 1.0, -1.0)

lam_p = 25.0
lam_e = 50.0
lam_l = 50.0
s_kwargs = {
    "split_choice" : "dyn_all_near_max",
    # "split_choice" : "all_max",
    # "split_choice" : "all_near_max",
    "pred_kind" : "prob",
    "slip" : 0.25,
    "w_path_slip" : True,
}


models = {
    # "stand" : {"model": STANDClassifier(**s_kwargs), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
    # "stand_active" : {"model": STANDClassifier(**s_kwargs), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn, "active_lrn" : True},
    # "stand_nos" : {"model": STANDClassifier(**s_kwargs, w_path_slip=False), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},

    # "stand_p" : {"model": STANDClassifier(**s_kwargs, lam_p=lam_p), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
    # "stand_p_e" : {"model": STANDClassifier(**s_kwargs, lam_p=lam_p, lam_e=lam_e), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
    # "stand_p_l" : {"model": STANDClassifier(**s_kwargs, lam_p=lam_p, lam_l=lam_l), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
    # "stand_e_l" : {"model": STANDClassifier(**s_kwargs, lam_e=lam_e, lam_l=lam_l), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
    
    
    "stand_p_e_l" : {"model": STANDClassifier(**s_kwargs, lam_p=lam_p, lam_e=lam_e, lam_l=lam_l), 
        "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},

    "stand_p_e_l_active" : {"model": STANDClassifier(**s_kwargs, lam_p=lam_p, lam_e=lam_e, lam_l=lam_l), 
        "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn, "active_lrn" : True},

    
    # "stand_w_slip" : {"model": STANDClassifier(**s_kwargs, lam_p=lam_p, lam_e=lam_e, lam_l=lam_l, w_path_slip=True),
    #     "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},

    # "stand_p5_e_l" : {"model": STANDClassifier(**s_kwargs, lam_p=5.0, lam_e=10.0, lam_l=lam_l), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
    # "stand_p10_e_l" : {"model": STANDClassifier(**s_kwargs, lam_p=10.0, lam_e=lam_e, lam_l=lam_l), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
    # "stand_p25_e_l" : {"model": STANDClassifier(**s_kwargs, lam_p=25.0, lam_e=lam_e, lam_l=lam_l), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
    # "stand_p50_e_l" : {"model": STANDClassifier(**s_kwargs, lam_p=50.0, lam_e=lam_e, lam_l=lam_l), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},

    # "stand_p_e5_l" : {"model": STANDClassifier(**s_kwargs, lam_p=lam_p, lam_e=5.0, lam_l=lam_l), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
    # "stand_p_e10_l" : {"model": STANDClassifier(**s_kwargs, lam_p=lam_p, lam_e=10.0, lam_l=lam_l), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
    # "stand_p_e25_l" : {"model": STANDClassifier(**s_kwargs, lam_p=lam_p, lam_e=25.0, lam_l=lam_l), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
    # "stand_p_e50_l" : {"model": STANDClassifier(**s_kwargs, lam_p=lam_p, lam_e=50.0, lam_l=lam_l), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
    # "stand_p_e100_l" : {"model": STANDClassifier(**s_kwargs, lam_p=lam_p, lam_e=100.0, lam_l=lam_l), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},

    # "stand_p_e_l5" : {"model": STANDClassifier(**s_kwargs, lam_p=lam_p, lam_e=lam_e, lam_l=5.0), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
    # "stand_p_e_l10" : {"model": STANDClassifier(**s_kwargs, lam_p=lam_p, lam_e=lam_e, lam_l=10.0), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
    # "stand_p_e_l25" : {"model": STANDClassifier(**s_kwargs, lam_p=lam_p, lam_e=lam_e, lam_l=25.0), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
    # "stand_p_e_l50" : {"model": STANDClassifier(**s_kwargs, lam_p=lam_p, lam_e=lam_e, lam_l=50.0), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
    # "stand_p_e_l100" : {"model": STANDClassifier(**s_kwargs, lam_p=lam_p, lam_e=lam_e, lam_l=100.0), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},

    # "stand_sl0" : {"model": STANDClassifier(**s_kwargs, lam_p=lam_p, lam_e=lam_e, lam_l=lam_l, slip=0.0), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
    # "stand_sl5" : {"model": STANDClassifier(**s_kwargs, lam_p=lam_p, lam_e=lam_e, lam_l=lam_l, slip=0.05), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
    # "stand_sl10" : {"model": STANDClassifier(**s_kwargs, lam_p=lam_p, lam_e=lam_e, lam_l=lam_l, slip=0.10), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
    # "stand_sl15" : {"model": STANDClassifier(**s_kwargs, lam_p=lam_p, lam_e=lam_e, lam_l=lam_l, slip=0.15), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
    # "stand_sl20" : {"model": STANDClassifier(**s_kwargs, lam_p=lam_p, lam_e=lam_e, lam_l=lam_l, slip=0.20), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
    # "stand_sl25" : {"model": STANDClassifier(**s_kwargs, lam_p=lam_p, lam_e=lam_e, lam_l=lam_l, slip=0.25), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
    # "stand_sl30" : {"model": STANDClassifier(**s_kwargs, lam_p=lam_p, lam_e=lam_e, lam_l=lam_l, slip=0.3), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
    # "stand_sl40" : {"model": STANDClassifier(**s_kwargs, lam_p=lam_p, lam_e=lam_e, lam_l=lam_l, slip=0.4), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
    # "stand_sl50" : {"model": STANDClassifier(**s_kwargs, lam_p=lam_p, lam_e=lam_e, lam_l=lam_l, slip=0.5), "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},

    # "xg_boost" : {"model": XGBClassifier(), "is_stand" : False, "one_hot" : True, "cert_fn" : xg_cert_fn},
    # "xg_boost_active" : {"model": XGBClassifier(), "is_stand" : False, "one_hot" : True, "cert_fn" : xg_cert_fn, "active_lrn" : True},
    # "random_forest" : {"model": RandomForestClassifier(), "is_stand" : False, "one_hot" : True, "cert_fn" : rf_cert_fn},
    # "decision_tree" : {"model": DecisionTreeClassifier(), "is_stand" : False, "one_hot" : True, "cert_fn" : dt_cert_fn },
    # "stand_dyn" : {"model": STANDClassifier(split_choice="dyn_all_near_max", lam_p=lam_l, lam_e=lam_e, lam_l=lam_l, pred_kind="probs"),
         # "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
    # "stand_near" : {"model": STANDClassifier(split_choice="all_near_max", lam_p=lam_l, lam_e=lam_e, lam_l=lam_l, pred_kind="probs"),
    #      "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
    
    # "stand_max" : {"model": STANDClassifier(split_choice="all_max", lam_p=lam_p, lam_e=lam_e, lam_l=lam_l, pred_kind="probs"),
    #      "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},

    
    
    # "stand_leafs" : {"model": STANDClassifier(**s_kwargs, lam_p=lam, lam_e=lam, pred_kind="max_leaves"),
    #      "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
    # "stand_density" : {"model": STANDClassifier(**s_kwargs, lam_p=lam, lam_e=lam, pred_kind="density"),
    #      "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
    # "stand_prob" : {"model": STANDClassifier(**s_kwargs, lam_p=lam, lam_e=lam, pred_kind="prob"),
    #      "is_stand" : True, "one_hot" : False, "cert_fn" : stand_cert_fn},
}

@njit(cache=True)
def mapback_ind_mask_subset(mask, inds):
    out_inds = -np.ones(len(inds), dtype=np.int64)
    for j, ind in enumerate(inds):

        k = 0
        for i, tf in enumerate(mask):
            if(tf):
                if(ind <= k):
                    out_inds[j] = i
                    break
                k +=1
            else:
                continue
        # out_inds[j] = -1
    return out_inds

# NOTES:
# 
def train_gen(X_train, y_train, incr, n_train=None, 
              active_lrn_model=None, is_stand=False):    
    if(n_train is None):
        n_train = len(X_train)

    if(not isinstance(incr, bool)):
        rng = range(0, n_train, incr)
    else:
        rng = range(0, n_train) if(incr) else [n_train]

    if(active_lrn_model is not None):
        not_covered = np.ones(len(X_train), dtype=np.bool_)
        X_train_buff = np.zeros((n_train, X_train.shape[1]), dtype=np.int32)
        y_train_buff = np.zeros(n_train, dtype=np.int32)
    #     print("ACTIVE")
    # else:
    #     print("NOT ACTIVE")

    for i in rng:
        end = i+int(incr)

        if(active_lrn_model is None):
            X_train_slc = X_train[0:end]
            y_train_slc = y_train[0:end]
            yield X_train_slc, y_train_slc
        else:
            if(i >= 2):
                if(is_stand):
                    probs, labels = active_lrn_model.predict_cert(X_train[not_covered], None)
                else:
                    probs = active_lrn_model.predict_proba(X_train[not_covered])

                max_probs = np.max(probs, axis=1)
                # pos_probs = probs[:,0]
                min_inds = np.argsort(max_probs)[:int(incr)]
                min_inds = mapback_ind_mask_subset(not_covered, min_inds)

                if(is_stand):
                    print()
                    print("v")
                    print(np.sort(max_probs))
                    min_10 = mapback_ind_mask_subset(not_covered, np.argsort(max_probs)[:10])
                    print("y", y_train[min_10])
                    active_lrn_model.bloop(X_train[min_10], None)
                    print(active_lrn_model)

                    conds = active_lrn_model.get_conds(1)
                    for conj in conds:
                        cs = []
                        for lit_option in conj:
                            ss = []
                            for lit in lit_option:
                                ss.append(f"[{lit[1]}]{'!=' if lit[0] else '=='}{lit[2]}")
                            cs.append(f"({' | '.join(ss)})")
                        print(", ".join(cs))

                    # print("^")
                    # active_lrn_model.bloop(X_train[np.argsort(-max_probs)[:int(incr)]], None)
                # print("<<", probs)
                # print(">>", np.sort(max_probs)[:int(incr)])
                
                # print(np.sort(max_probs)[:int(incr)])
                # print("<<", np.all(not_covered[min_inds]))
                not_covered[min_inds] = False
            else:
                min_inds = np.arange(i,end)

            X_train_buff[i:end] = X_train[min_inds]
            y_train_buff[i:end] = y_train[min_inds]

            X_train_slc = X_train_buff[0:end]
            y_train_slc = y_train_buff[0:end]

            # ex_mask = y_train_slc!=1.0
            # X_train_slc = np.concatenate([X_train_slc, X_train_slc[ex_mask]])
            # y_train_slc = np.concatenate([y_train_slc, y_train_slc[ex_mask]])

            yield X_train_slc, y_train_slc



def test_model(name, config, data, one_hot_encoder, 
                incr=True, n_train=100, 
                is_stand=False, calc_certs=False):
    X_train, X_test, y_train, y_test = data

    model = config['model']
    is_stand = config['is_stand']
    one_hot = config['one_hot']
    cert_fn = config['cert_fn']
    active_lrn = config.get('active_lrn', False)
    holdout_certs = []
    holdout_accuracies = []

    # One-hot encode models that aren't stand (maybe should get rid of this
    #   the issue is that some of the models don't quite work without one-hot)
    if(one_hot):
        X_train = one_hot_encoder.transform(X_train).toarray()
        X_test = one_hot_encoder.transform(X_test).toarray()
    
    
    active_lrn_model = model if active_lrn else None
    tg = train_gen(X_train, y_train, incr, n_train, 
                    active_lrn_model, is_stand)

    for X_train_slc, y_train_slc in tg:
        # # Fit on subset of training set 
        # X_train_slc = X_train[0:i]
        # y_train_slc = y_train[0:i]

        # with PrintElapse(f"fit {name}"):
        if(is_stand):
            model.fit(X_train_slc.astype(dtype=np.int32), None, y_train_slc)
            # print(model)
        else:
            model.fit(X_train_slc, y_train_slc)

        # with PrintElapse(f"preict {name}"):
        # Test on the test_set
        if(is_stand):
            y_preds = model.predict(X_test, None)
        else:
            y_preds = model.predict(X_test)



        corrs = y_preds == y_test


        #############################
        # print("--------------------------------")
        # fails = X_test[~corrs]
        # print("BEFORE:", accuracy_score(y_test, y_preds))
        # probs, labels = model.predict_proba(X_test, None)
        # best_ind = np.argmax(probs, axis=1)
        # new_y_preds = labels[best_ind]
        # print("AFTER:", accuracy_score(y_test, new_y_preds))

        # model.predict(X_test[:1], None)
        # model.predict_proba(X_test[24:26], None)

        # print("--------------------------------")
        # model.predict_proba(fails, None)
        # raise ValueError()
        # print("--------------------------------")
        #############################

        if(cert_fn and calc_certs):
            y_certs = cert_fn(model, X_test)

            # print("y_certs", y_certs)
            holdout_certs.append(y_certs)
        holdout_accuracies.append(accuracy_score(y_test, y_preds))



    # print("FP_reocc:", stats["FP_reocc"])
    # print("FN_reocc:", stats["FN_reocc"])
    # print("error_reocc:", stats["error_reocc"])
        
    # if(name == "stand_p"):
    #     print(model)
    print()
    print(name)

    stats = {"model_name" : name,
             "accuracies" :  holdout_accuracies,
             "accuracy" : holdout_accuracies[-1],
            }
    # print("Accuracies: ", holdout_accuracies)
    # print("Accuracy@10: ", holdout_accuracies[int(10/incr)])
    print("Accuracy@20: ", holdout_accuracies[int(20/incr)])
    print("Accuracy@50: ", holdout_accuracies[int(50/incr)])
    # print("Accuracy   : ", holdout_accuracies[-1])
    
    if(cert_fn and calc_certs):

        cert_stats = eval_total_cert_stats(y_test==1.0, holdout_certs)
        stats.update(cert_stats)

        
        #print("total_error_reocc:", stats["total_error_reocc"])
        # print("prod_monot:", stats["prod_monot"])
        #print("w_prod_monot:", stats["w_prod_monot"])

        print("total_prod_monot:", stats["total_prod_monot"])
        print("total_w_prod_monot:", stats["total_w_prod_monot"])
        # print("total_FP_reocc:", stats["total_FP_reocc"])
        # print("total_FN_reocc:", stats["total_FN_reocc"])

        avg_abs_prec_res = 0.0
        n_preds = 0.0

        for cert_bin in DEFAULT_CERT_BINS:
            c_min, c_max = cert_bin
            c_mean = (c_min + c_max) / 2
            c_hrng = (c_max - c_min) / 2
            prec = stats[('total_precision', cert_bin)]
            TP_n = stats[('TP_n', cert_bin)]
            bin_n = stats[('bin_n', cert_bin)]
            # print(f"total_precision @ {100*c_mean:.1f} +/- {100*c_hrng:.1f}: {prec:.2f} {TP_n}/{bin_n}")
            print(f"precision_res @ {100*c_mean:.1f} +/- {100*c_hrng:.1f}: {prec-c_mean:.2f} {TP_n}/{bin_n}")
            # print("total_precision @ 1.0:", stats[("total_precision",1.0)])

            avg_abs_prec_res += np.abs((prec-c_mean)) * bin_n
            n_preds += bin_n
        avg_abs_prec_res /= n_preds
        stats['avg_abs_prec_res'] = avg_abs_prec_res

        print(f"avg_abs_prec_res:", avg_abs_prec_res)


    return stats




from numba import njit
import numpy as np

@njit(cache=True)
def index_of(array, item):
    for idx, val in enumerate(array):
        if val == item:
            return idx

def ensure_first_neg_pos(X_train, y_train):
    first_0 = index_of(y_train, 0)
    # print("first_0", first_0)
    X_train[[0, first_0]] = X_train[[first_0, 0]]
    y_train[[0, first_0]] = y_train[[first_0, 0]]

    first_1 = index_of(y_train, 1)

    # Swap
    
    X_train[[1, first_1]] = X_train[[first_1, 1]]
    y_train[[1, first_1]] = y_train[[first_1, 1]]

    return X_train, y_train

dnf = None
def gen_data(n_train=200, n_test=2000):
    global dnf

    one_hot_encoder = OneHotEncoder()

    min_one_possion = lambda x : 1+poisson(x-1)
    min_two_possion = lambda x : 2+poisson(x-2)


    # np.random.seed()
    

    X, y, dnf = gen_synthetic_dnf_data(
                            n_samples=n_train + n_test,
                            n_feats=400,
                            vals_per_feat= lambda : min_two_possion(3),
                            pos_prop=.5,

                            # conj_len= lambda : min_one_possion(2), 
                            # num_conj= lambda : min_one_possion(2),
                            conj_len=2,
                            num_conj=2,
                            dupl_lit_prob=0.3,
                            conj_probs=.28,

                            neg_conj_len=lambda : min_two_possion(4),
                            num_neg_conj=100,
                            neg_dupl_lit_prob=0.1,
                            neg_conj_probs=.8,

                            force_same_vals=False)


    print_dnf(dnf)
    y[y!=1] = 0
    one_hot_encoder.fit(X)

    data = biased_train_test_split(X, y, train_size=n_train, true_prop=.8)

    X_train, X_test, y_train, y_test = data

    #print("y_train", y_train)
    # print("=1 Prop", np.sum(y_train==1)/len(y_train), np.sum(y_test==1)/len(y_test))

    # raise ValueError()
    # X_train, y_train = front_load_neg(X_train, y_train)
    X_train, y_train = ensure_first_neg_pos(X_train, y_train)

    data = (X_train, X_test, y_train, y_test)
    return data, one_hot_encoder


def run_and_save_stats(models):
    time = datetime.now().strftime('%m-%d-%Y_%H:%M:%S')

    data, one_hot_encoder = gen_data();

    stats_by_model = {}
    for i, (name, config) in enumerate(models.items()):
        is_stand = name == "stand"
        stats = test_model(name, config, data, one_hot_encoder, 
                    incr=1, is_stand=is_stand, calc_certs=True)
        print_dnf(dnf)
        stats_by_model[name] = stats

    os.makedirs("sim_saves", exist_ok=True)    
    with open(f"sim_saves/run_{time}", 'wb+') as f:
        pickle.dump(stats_by_model, f, protocol=pickle.HIGHEST_PROTOCOL)


# seed = int(10000*py_random())
    # seed = 5545
# # seed = 4855
# seed = 8931
# np.random.seed(seed)
# seeds = np.arange(100)
for i in range(1):

    # seed = int(10000*py_random())
    # seed = 5545
    # seed = 4855
    # seed = 8931
    np.random.seed(i+23)
    print("------------------------------------")
    run_and_save_stats(models)
# X_one_hot = one_hot_encoder.transform(X).toarray()

# print("X")
# print(X)















# print(X)
# print(y)

