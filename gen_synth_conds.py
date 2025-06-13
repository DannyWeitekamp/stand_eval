import numpy as np
from numpy.random import random
import warnings
from copy import copy


def print_dnf(dnf):
    for conj in dnf:
        cs = ", ".join([f"({lit['ind']}, {lit['val']})" for lit in conj])
        print(cs)

conj_type = [('ind', 'i8'), ('val', 'i8')]

def random_conj(data, conj_len, 
                other_conjs=[], # Note: This basically 
                dupl_lit_prob=0.0,
                force_same_vals=False):

    conj = np.zeros(conj_len, dtype=conj_type)

    # Randomly pick one row in the data and a set of feature indicies
    feat_inds = np.random.choice(np.arange(data.shape[1]), size=conj_len, replace=False)
    feat_inds = np.sort(feat_inds)
    rand_int = np.random.randint(len(data))
    row = data[rand_int]
    feat_vals = row[feat_inds].copy()

    # Make a conjuction from the random feature indicies 
    conj['ind'] = feat_inds
    conj['val'] = feat_vals

    # With dupl_lit_prob replace literals in the current conjunction 
    #  with ones from previous conjunctions in the dnf.
    if(len(other_conjs) > 0 and dupl_lit_prob > 0.0):
        # Find all previous unique literals
        all_conj = np.concatenate(other_conjs)
        unq_conj = np.unique(all_conj)
        
        # Randomly selection which literals will be replaced
        dup_mask = np.random.random(conj_len) < dupl_lit_prob
        dupl_inds = np.nonzero(dup_mask)[0]
        nodupl_inds = np.nonzero(~dup_mask)[0]

        if(len(dupl_inds) > 0):
            # Make it impossible to select feature indicies from unq_conj 
            #  that would cause a contradiction in a conjunction
            forbidden_lit_inds = feat_inds[nodupl_inds]
            unq_conj = unq_conj[~np.isin(unq_conj['ind'], forbidden_lit_inds)]
            
            
            # Randomly selection the literals and replace them
            dupl_conjs = np.random.choice(unq_conj, size=min(len(dupl_inds), len(unq_conj)), replace=False)
            conj[dupl_inds] = dupl_conjs

    # Sort inds by increasing  
    conj = conj[np.argsort(conj['ind'])]
    
    # If set to true then any literal in the dnf with the same feature 
    #   index must also check for the same value (probably never need this).
    if(force_same_vals):
        other_vals = np.zeros(data.shape[1], dtype=np.int64)
        for conj in other_conjs:
            for lit in conj:
                other_vals[lit['ind']] = lit['val']

        ov = other_vals[feat_inds]
        feat_vals = np.where(ov == 0, feat_vals, ov)
    else:
        feat_vals = row[feat_inds]
    
    return conj

def random_dnf(data, conj_len, num_conj, dupl_lit_prob, 
    force_same_vals,
    prev_conjs=[]):
    prev_conjs = copy(prev_conjs)
    nc = num_conj() if hasattr(num_conj, "__call__") else num_conj
    dnf = []
    for i in range(nc):
        cl = conj_len() if hasattr(conj_len, "__call__") else conj_len
        conj =random_conj(data, cl, prev_conjs, dupl_lit_prob, force_same_vals)
        dnf.append(conj)
        prev_conjs.append(conj)
    return dnf

def random_uniform_data(n_samples, n_feats, n_feat_bound=6):
    #print(vals_per_feat)
    data = np.random.randint(1, n_feat_bound, size=(n_samples, n_feats))
    return data

def ensure_violates(x, conj, n_feat_bound, prob=.1, near_violate=False):
    if(near_violate):
        x[conj['ind']] = conj['val']
        # print("MID:", x[conj['ind']], conj['val'])
    # print("ensure_violates", conj[0])
    inds = conj['ind']
    
    subset = [lit for lit in conj if random() < prob]
    if(len(subset) == 0):
        subset = [np.random.choice(conj)]

    for lit in subset:
        ind = lit['ind']
        new_val = bad_val = lit['val']
        while(new_val == bad_val):
            #print(n_feat_bound[0, ind])
            if(isinstance(n_feat_bound, np.ndarray)):
                new_val = np.random.randint(1, n_feat_bound[0, ind])
            else:
                new_val = np.random.randint(1, n_feat_bound)
        x[ind] = new_val
    # print("end")



def structure_and_label_data(X, y, dnf, label=1, 
    conj_probs=.2,
    n_feat_bound=None,
    modified_data_mask=None,
    modified_feat_mask=None,
    prevent_overlap=True,

    ):
    if(isinstance(conj_probs,float)):
        conj_probs = (conj_probs,)*len(dnf)

    assert len(conj_probs) == len(dnf), f"conj_probs {conj_probs} must match number of conjunctions"

    new_X = X.copy()
    new_y = y.copy()
    
    if(modified_data_mask is None):
        modified_data_mask = np.zeros(X.shape[0], dtype=np.bool_)
    if(modified_feat_mask is None):
        modified_feat_mask = np.zeros(X.shape[1], dtype=np.bool_)

    lst_candidates, lst_satisfied_inds = [], []
    for k, conj in enumerate(dnf):
        # Find rows in the data that already satisfy "conj"
        #  or were modified by a previous one.
        n_samples = int(conj_probs[k]*len(new_X))
        lit_mask = new_X[:, conj['ind']] == conj['val']
        satisfied_mask = np.all(lit_mask, axis=1)
        satisfied_inds = np.nonzero(satisfied_mask)[0]

        # If too many samples already violate the conj then resample
        #  some of their values
        if(len(satisfied_inds) > n_samples):
            v_inds = np.random.choice(satisfied_inds, 
                size=len(satisfied_inds)-n_samples)

            for ind in v_inds:
                ensure_violates(X[ind, :], conj, n_feat_bound)    
                satisfied_mask[ind] = False                

            satisfied_inds = np.nonzero(satisfied_mask)[0]

        lst_satisfied_inds.append(satisfied_inds)

        # Skip data points that have been modified by a previous conjunction
        #  and touch the same features as a previous conjunction
        touch_prev_feat = np.any(modified_feat_mask[conj['ind']])
        skip_inds = np.nonzero((touch_prev_feat & modified_data_mask) | satisfied_mask)[0]
        
        # Sample a set of data points to modify (that are not already satisfied)
        candidates = np.delete(np.arange(new_X.shape[0]), skip_inds)
        lst_candidates.append(candidates)

    
    for k, (conj, satisfied_inds, candidates) in enumerate(
            zip(dnf, lst_satisfied_inds, lst_candidates)
        ):
        n_to_modify = min(max(0, n_samples-len(satisfied_inds)), len(candidates))

        if(n_to_modify < n_samples-len(satisfied_inds)):
            warnings.warn("A conjunct cannot make sufficient data modifications. " +
                "Try lowering conj_probs.")

        # print("n_to_modify", len(candidates), n_to_modify, n_samples-len(satisfied_inds))

        try:
            mod_inds = np.random.choice(candidates, size=n_to_modify, replace=False)

            if(prevent_overlap):
                modified_data_mask[mod_inds] = True
                modified_feat_mask[conj['ind']] = True

            # Modify the previously random samples so they satisfy 'conj'
            for ind in mod_inds:
                new_X[ind, conj['ind']] = conj['val']

            # Assign a positive label '1' to the satisfied and modified data
            new_y[satisfied_inds] = label+k if label == 2 else label
            new_y[mod_inds] = label+k if label == 2 else label
        except Exception:
            pass
    return new_X, new_y

def gen_synthetic_dnf_data(n_samples,
                          n_feats,
                          vals_per_feat,
                          pos_prop=.5,

                          conj_len=4, 
                          num_conj=3,
                          dupl_lit_prob=0.2,
                          conj_probs=.3,

                          neg_conj_len=10,
                          num_neg_conj=20,
                          neg_dupl_lit_prob=0.4,
                          neg_conj_probs=.2,
                          
                          force_same_vals=False,
                          neg_violate=False,
                          neg_near_violate=False):
    
    if hasattr(vals_per_feat, "__call__"):
        n_feat_bound = np.reshape([vals_per_feat()+1 for _ in range(n_feats)], (1,n_feats))
    else:
        n_feat_bound = vals_per_feat + 1
    # print(vals_per_feat.shape)
    X = random_uniform_data(n_samples, n_feats, n_feat_bound)
    y = np.zeros(len(X))
    dnf = random_dnf(X, conj_len, num_conj,
            dupl_lit_prob, force_same_vals)
    neg_dnf = random_dnf(X, neg_conj_len, num_neg_conj,
            neg_dupl_lit_prob, force_same_vals, prev_conjs=dnf)

    # neg_dnfs = random_dnf(X, conj_len, num_neg_conj, dupl_lit_prob, force_same_vals)

    modified_data_mask = np.zeros(X.shape[0], dtype=np.bool_)
    modified_feat_mask = np.zeros(X.shape[1], dtype=np.bool_)

    X, y = structure_and_label_data(X, y, neg_dnf, 2, neg_conj_probs, n_feat_bound,
        modified_data_mask, modified_feat_mask, prevent_overlap=False)

    modified_data_mask = np.zeros(X.shape[0], dtype=np.bool_)
    modified_feat_mask = np.zeros(X.shape[1], dtype=np.bool_)

    X, y = structure_and_label_data(X, y, dnf, 1, conj_probs, n_feat_bound,
        modified_data_mask, modified_feat_mask)

    if(neg_violate or neg_near_violate):
        # print("DO IT", y[:10])
        for ind in range(n_samples):
            if(y[ind] != 1):
                for conj in dnf:
                    # print("BEFORE:", X[ind, conj['ind']], conj['val'])
                    ensure_violates(X[ind], conj, n_feat_bound,
                        near_violate=neg_near_violate)
                    # print("AFTER :", X[ind, conj['ind']])

    
    # y[y==0] = -1 # Make the negative class -1
    return X, y, dnf



if __name__ == "__main__":
    from numpy.random import poisson
    min_one_possion = lambda x : 1+poisson(x-1)
    min_two_possion = lambda x : 2+poisson(x-2)

    X, y, dnf = gen_synthetic_dnf_data(
                          n_samples=2200,
                          n_feats=50,
                          vals_per_feat= lambda : min_two_possion(3),
                          pos_prop=.5,

                          conj_len= lambda : min_one_possion(1), 
                          num_conj= lambda : min_two_possion(2),
                          dupl_lit_prob=0.3,
                          conj_probs=.28,

                          neg_conj_len=lambda : min_two_possion(3),
                          num_neg_conj=100,
                          neg_dupl_lit_prob=0.1,
                          neg_conj_probs=.8,

                          force_same_vals=False)





    #print(X)
    #print(y)
    print("=1 Prop", np.sum(y==1)/len(y))
    print_dnf(dnf)
