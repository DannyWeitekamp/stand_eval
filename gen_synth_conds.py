import numpy as np
import warnings


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

        # Make it impossible to select feature indicies from unq_conj 
        #  that would cause a contradiction in a conjunction
        forbidden_lit_inds = feat_inds[nodupl_inds]
        unq_conj = unq_conj[~np.isin(unq_conj['ind'], forbidden_lit_inds)]
        
        # Randomly selection the literals and replace them
        dupl_conjs = np.random.choice(unq_conj, size=len(dupl_inds), replace=False)
        conj[dupl_inds] = dupl_conjs
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

def random_dnf(data, conj_len, num_conj, dupl_lit_prob, force_same_vals):
    dnf = []
    for i in range(num_conj):
        dnf.append(random_conj(data, conj_len, dnf, dupl_lit_prob, force_same_vals))
    return dnf

def random_uniform_data(n_samples, n_feats, vals_per_feat=5):
    data = np.random.randint(1, vals_per_feat, size=(n_samples, n_feats))
    return data

def structure_and_label_data(data, dnf, conj_probs=.2):
    if(isinstance(conj_probs,float)):
        conj_probs = (conj_probs,)*len(dnf)

    assert len(conj_probs) == len(dnf), f"conj_probs {conj_probs} must match number of conjunctions"

    new_data = data.copy()
    y = np.zeros(len(data))

    modified_data_mask = np.zeros(data.shape[0], dtype=np.bool_)
    modified_feat_mask = np.zeros(data.shape[1], dtype=np.bool_)
    for k, conj in enumerate(dnf):
        # Find rows in the data that already satisfy "conj"
        #  or were modified by a previous one.
        lit_mask = new_data[:, conj['ind']] == conj['val']
        satisfied_mask = np.all(lit_mask, axis=1)
        
        # Skip data points that have been modified by a previous conjunction
        #  and touch the same features as a previous conjunction
        touch_prev_feat = np.any(modified_feat_mask[conj['ind']])
        skip_inds = np.nonzero((touch_prev_feat & modified_data_mask) | satisfied_mask)[0]
        satisfied_inds = np.nonzero(satisfied_mask)[0]

        # Sample a set of data points to modify (that are not already satisfied)
        n_samples = int(conj_probs[k]*len(new_data))
        candidates = np.delete(np.arange(new_data.shape[0]), skip_inds)
        n_to_modify = min(max(0, n_samples-len(satisfied_inds)), len(candidates))

        if(n_to_modify < n_samples-len(satisfied_inds)):
            warnings.warn("A conjunct cannot make sufficient data modifications. " +
                "Try lowering conj_probs.")

        print("n_to_modify", len(candidates), n_to_modify, n_samples-len(satisfied_inds))

        mod_inds = np.random.choice(candidates, size=n_to_modify, replace=False)
        modified_data_mask[mod_inds] = True
        modified_feat_mask[conj['ind']] = True

        # Modify the previously random samples so they satisfy 'conj'
        for ind in mod_inds:
            new_data[ind, conj['ind']] = conj['val']

        # Assign a positive label '1' to the satisfied and modified data
        y[satisfied_inds] = 1
        y[mod_inds] = 1
    return new_data, y

def gen_synthetic_dnf_data(n_samples,
                          n_feats,
                          vals_per_feat,
                          conj_len, 
                          num_conj=1,
                          conj_probs=.2,
                          dupl_lit_prob=0.0,
                          force_same_vals=False):
    X = random_uniform_data(n_samples, n_feats, vals_per_feat)    
    dnf = random_dnf(X, conj_len, num_conj, dupl_lit_prob, force_same_vals)
    X, y = structure_and_label_data(X, dnf, conj_probs)
    # y[y==0] = -1 # Make the negative class -1
    return X, y, dnf



if __name__ == "__main__":
    X, y, dnf = gen_synthetic_dnf_data(100, 100, 5, 5, 10, .1)

    print(X)
    print(y)
    print_dnf(dnf)
