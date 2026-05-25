import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_models import gen_data
from stand.stand import STANDClassifier

s_kwargs = {
    "split_choice": "dyn_all_near_max",
    "pred_kind": "prob",
    "slip": 0.1,
    "w_path_slip": True,
    "impurity_func": "gini",
    "gain_method": "impurity_decrease",
    "impurity_agg_method": "weighted_sum",
}

hs_params = {
    "lam_p" : 25.0,
    "lam_e" : 25.0,
    "lam_l" : 50.0,
}


def mislabel_uncertainty(probs, labels, held_out_label):
    """
    probs:           shape (1, n_classes) from predict_proba
    labels:          class label array returned alongside probs
    held_out_label:  the (possibly flipped) label of the held-out sample

    Returns:
        1 - max_prob  if the held-out label has the higher probability
        1 + max_prob  if the opposite label has the higher probability
    """
    probs_1d = probs[0]
    max_idx = int(np.argmax(probs_1d))
    max_prob = float(probs_1d[max_idx])
    max_label = labels[max_idx]

    if max_label == held_out_label:
        return 1.0 - max_prob
    else:
        return 1.0 + max_prob


def run_trial(seed):
    np.random.seed(seed)

    data, _ = gen_data()
    X_train, _, y_train, _ = data
    n = len(X_train)

    # Step 1: flip one randomly chosen sample's label
    flip_idx = np.random.randint(0, n)
    y_mod = y_train.copy()
    y_mod[flip_idx] = 1 - y_mod[flip_idx]

    # Step 2: leave-one-out refits
    uncertainties = np.empty(n)

    for i in range(n):
        loo_mask = np.ones(n, dtype=bool)
        loo_mask[i] = False

        X_loo = X_train[loo_mask].astype(np.int32)
        y_loo = y_mod[loo_mask]

        # STAND requires at least one sample of each class
        if len(np.unique(y_loo)) < 2:
            uncertainties[i] = 2.0  # maximum possible uncertainty
            continue

        model = STANDClassifier(**s_kwargs, **hs_params)
        model.fit(X_loo, None, y_loo)

        X_ho = X_train[i : i + 1].astype(np.int32)
        probs, labels = model.predict_proba(X_ho, None)

        uncertainties[i] = mislabel_uncertainty(probs, labels, y_mod[i])

    # Step 3: rank samples by descending uncertainty (rank 1 = most uncertain)
    ranked = np.argsort(uncertainties)[::-1]
    rank = int(np.where(ranked == flip_idx)[0][0]) + 1  # 1-indexed

    return rank


def main():
    N = 100
    ranks = []

    for trial in range(N):
        print(f"\n=== Trial {trial + 1}/{N} ===")
        rank = run_trial(seed=trial + 42)
        ranks.append(rank)
        print(f"  Flipped sample rank: {rank}")

    ranks = np.array(ranks)

    print("\n" + "=" * 40)
    print(f"Results over {N} trials")
    print("=" * 40)
    for k in range(1, 6):
        count = int(np.sum(ranks == k))
        pct = 100.0 * count / N
        print(f"  Rank {k}: {pct:5.1f}%  ({count}/{N})")
    top5 = int(np.sum(ranks <= 5))
    print(f"  Top-5 total: {100.0 * top5 / N:.1f}%  ({top5}/{N})")
    print("=" * 40)


if __name__ == "__main__":
    main()
