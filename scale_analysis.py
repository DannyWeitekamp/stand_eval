"""
Scale analysis: compare TreeClassifier (decision_tree), TreeClassifier (option_tree),
and STANDClassifier across multiple datasets and training set sizes.
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from stand.tree_classifier import TreeClassifier
from stand.stand import STANDClassifier

# ── Datasets to evaluate ──────────────────────────────────────────────────────
# Each entry: (openml_name, version)
DATASETS = [
    ("satimage",  1),
    ## ("mushroom",  1),
    ("letter",    1),
    ## ("adult",     2),
    # ("isolet", 1)
]

TRAIN_SIZES = [50, 100, 200, 400, 800, 1600, 3200, 6000, 12000, 18000]
# Normalize in case trailing comma wrapped the list in a tuple
if isinstance(TRAIN_SIZES, tuple):
    TRAIN_SIZES = [s for item in TRAIN_SIZES for s in (item if isinstance(item, list) else [item])]

N_REPEATS = 1
RNG_SEED = 42

# ── Dataset loading ───────────────────────────────────────────────────────────

def load_dataset(name, version):
    """
    Load an OpenML dataset and split into X_nom (int32), X_cont (float64), Y (int64).
    Nominal features are identified via data.categories (set by liac-arff parser).
    Missing values are imputed: median for continuous, mode for nominal.
    Returns (X_nom, X_cont, Y).
    """
    data = fetch_openml(name, version=version, as_frame=False, parser="liac-arff")
    X_raw = data.data
    feature_names = list(data.feature_names)
    cat_set = set(data.categories.keys())

    nom_cols  = [i for i, n in enumerate(feature_names) if n in cat_set]
    cont_cols = [i for i, n in enumerate(feature_names) if n not in cat_set]

    X_nom_raw  = X_raw[:, nom_cols]  if nom_cols  else np.empty((len(X_raw), 0))
    X_cont_raw = X_raw[:, cont_cols] if cont_cols else np.empty((len(X_raw), 0))

    # Impute missing values
    if nom_cols:
        for j in range(X_nom_raw.shape[1]):
            col = X_nom_raw[:, j]
            mask = np.isnan(col)
            if mask.any():
                vals, counts = np.unique(col[~mask], return_counts=True)
                mode = vals[np.argmax(counts)]
                col[mask] = mode

    if cont_cols:
        for j in range(X_cont_raw.shape[1]):
            col = X_cont_raw[:, j]
            mask = np.isnan(col)
            if mask.any():
                col[mask] = np.nanmedian(col)

    X_nom  = X_nom_raw.astype(np.int32)
    X_cont = X_cont_raw.astype(np.float64)

    Y = LabelEncoder().fit_transform(data.target).astype(np.int64)

    return X_nom, X_cont, Y


# ── Helpers ───────────────────────────────────────────────────────────────────

def count_nodes(model):
    if isinstance(model, STANDClassifier):
        return len(model.op_tree_classifier.nodes)
    return len(model.nodes)


lam_p = 25.0
lam_e = 25.0
lam_l = 50.0

CONDITIONS = [
    ("Dec. Tree", lambda: TreeClassifier(preset_type="decision_tree"), None),
    ("Opt. Tree",   lambda: TreeClassifier(preset_type="option_tree"),   None),
    # ("STAND (a=.0)",               lambda: STANDClassifier(slip=.0), None),
    # ("STAND (a=.1)",               lambda: STANDClassifier(slip=.1), None),
    ("STAND (a=.2)",               lambda: STANDClassifier(slip=.2), None),
    # ("S h.s. (a=.1)",               lambda: STANDClassifier(slip=.1, lam_p=lam_p, lam_e=lam_e, lam_l=lam_l), None),
]


def fmt(mean, std, _int=False):
    if(_int):
        return f"{int(mean)} ± {std:.1f}"
    else:
        return f"{mean:.1f} ± {std:.1f}"

def run_dataset(ds_name, ds_version, rng):
    print(f"\n{'='*80}")
    print(f"Dataset: {ds_name} (version {ds_version})")
    print(f"{'='*80}")

    X_nom, X_cont, Y = load_dataset(ds_name, ds_version)
    n_total = len(Y)
    print(f"  Loaded: {n_total} samples, {X_nom.shape[1]} nominal + {X_cont.shape[1]} continuous features")

    # Hold out 10% for test, rest is training pool
    n_test = max(1, int(0.1 * n_total))
    test_idx  = rng.choice(n_total, size=n_test, replace=False)
    train_mask = np.ones(n_total, dtype=bool)
    train_mask[test_idx] = False

    X_nom_test  = X_nom[test_idx]
    X_cont_test = X_cont[test_idx]
    Y_test      = Y[test_idx]

    X_nom_pool  = X_nom[train_mask]
    X_cont_pool = X_cont[train_mask]
    Y_pool      = Y[train_mask]
    n_pool      = len(Y_pool)

    print(f"  Test set: {len(Y_test)} samples | Pool: {n_pool} samples\n")

    # Only run sizes up to the pool size; cap actual sample at pool size
    valid_sizes = [n for n in TRAIN_SIZES if n <= n_pool]
    if(valid_sizes[-1] != n_pool):
        valid_sizes += [n_pool]
    # If even the smallest size exceeds the pool, still run on whatever we have
    if not valid_sizes:
        valid_sizes = [TRAIN_SIZES[0]]

    # results[condition][train_size] = {"acc": [], "nodes": []}
    results = {name: {n: {"acc": [], "nodes": []} for n in valid_sizes}
               for name, *_ in CONDITIONS}

    total = len(valid_sizes) * N_REPEATS * len(CONDITIONS)
    done = 0

    for train_size in valid_sizes:
        actual_size = min(train_size, n_pool)
        for rep in range(N_REPEATS):
            if actual_size == n_pool:
                # Use all pool data — no randomness, so only run once
                idx = np.arange(n_pool)
            else:
                idx = rng.choice(n_pool, size=actual_size, replace=False)
            X_nom_tr  = X_nom_pool[idx]
            X_cont_tr = X_cont_pool[idx]
            Y_tr      = Y_pool[idx]

            for name, make_model, max_train in CONDITIONS:
                if max_train is not None and len(Y_tr) > max_train:
                    # cap_idx = rng.choice(len(Y_tr), size=max_train, replace=False)
                    # fit_nom, fit_cont, fit_y = X_nom_tr[cap_idx], X_cont_tr[cap_idx], Y_tr[cap_idx]
                    if(train_size > max_train):
                        continue
                else:
                    fit_nom, fit_cont, fit_y = X_nom_tr, X_cont_tr, Y_tr

                print(f"\r [{done+1}/{total}] {name} n={actual_size} rep={rep+1}", end="", flush=True)
                model = make_model()
                model.fit(fit_nom, fit_cont, fit_y)
                preds = model.predict(X_nom_test, X_cont_test)
                acc = np.mean(preds == Y_test)
                # print(model)
                n_nodes = count_nodes(model)
                results[name][train_size]["acc"].append(acc)
                results[name][train_size]["nodes"].append(n_nodes)
                done += 1
                print(f" | acc={acc:.3f} nodes={n_nodes}", end="", flush=True)

            if actual_size == n_pool:
                break  # No point repeating identical data

    # Print tables
    col_w = 22
    for metric, key in [("Test Accuracy", "acc"), ("Number of Nodes", "nodes")]:
        avg_kind = "median" if(metric == "Number of Nodes") else "mean"

        print(f"\n[ {metric} ({avg_kind} ± std over {N_REPEATS} random subsets) ]\n")
        header = f"{'Train size':>12}" + "".join(f"{n:>{col_w}}" for n, *_ in CONDITIONS)
        print(header)
        print("-" * len(header))
        for n in valid_sizes:
            row = f"{n:>12}"
            for name, *_ in CONDITIONS:
                vals = results[name][n][key]
                if(len(vals) == 0):
                    row += f"{'---':>{col_w}}"
                else:
                    if(avg_kind == "median"):
                        row += f"{fmt(np.median(vals), np.std(vals), 1):>{col_w}}"
                    else:
                        row += f"{fmt(np.mean(vals)*100, np.std(vals)):>{col_w}}"
            print(row)


# ── Main ──────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(RNG_SEED)

for ds_name, ds_version in DATASETS:
    run_dataset(ds_name, ds_version, rng)

print("\nDone.")
