"""
Comparison of XGBoost, DecisionTree, and RandomForest on UCI binary classification datasets
with categorical features.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, ShuffleSplit
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.datasets import fetch_openml
from stand.stand import STANDClassifier as _STANDClassifier

lam_p = 25.0
lam_e = 25.0
lam_l = 50.0

shared_kwargs = {
    "split_choice" : "dyn_all_near_max",
    # "split_choice" : "all_max",
    # "split_choice" : "all_near_max",
    "pred_kind" : "prob",
    "slip" : 0.1,
    "w_path_slip" : True,
}

s_kwargs = {
    **shared_kwargs,
    "impurity_func" : "gini",
    "gain_method" : "impurity_decrease",
    # "gain_method" : "foil",
    # "impurity_agg_method" : "most_positive"
    "impurity_agg_method" : "weighted_sum"
}

class STANDClassifier(_STANDClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split_choice = "dyn_all_near_max"
        self.pred_kind = "probs"
    
    def fit(self, X, y):
        super().fit(X, None, y)
        return self
    
    def predict(self, X):
        return super().predict(X, None)
    
    def predict_proba(self, X):
        return super().predict_proba(X, None)
try:
    import openml
    HAS_OPENML = True
except ImportError:
    HAS_OPENML = False
import warnings
warnings.filterwarnings('ignore')

def load_uci_dataset(name, target_col=None):
    """
    Load a UCI dataset from OpenML.
    
    Parameters:
    -----------
    name : str
        Dataset name or OpenML ID
    target_col : str, optional
        Target column name if known
    
    Returns:
    --------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    """
    print(f"\n{'='*60}")
    print(f"Loading dataset: {name}")
    print(f"{'='*60}")
    
    try:
        # Try to fetch by name first
        data = fetch_openml(name=name, version=1, as_frame=True, parser='auto')
        X = data.data
        y = data.target
        
        # If target is not binary, try to find a binary target or skip
        # if y.nunique() > 2:
        #     print(f"Warning: Dataset has {y.nunique()} classes, skipping...")
        #     return None, None
        

        print(f"Dataset shape: {X.shape}")
        print(f"Target distribution:\n{y.value_counts()}")
        print(f"Features: {list(X.columns)}")
        print(f"Majority Class Baseline: {np.max(y.value_counts())/np.sum(y.value_counts()):.2f}",  )
        
        return X, y
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None, None

def preprocess_categorical(X, y):
    """
    Preprocess categorical features and target.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    
    Returns:
    --------
    X_encoded : np.ndarray
        Encoded feature matrix
    y_encoded : np.ndarray
        Encoded target vector
    feature_names : list
        List of feature names after encoding
    """
    # Encode target
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
    # Encode categorical features
    if categorical_cols:
        ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        X_cat_encoded = ohe.fit_transform(X[categorical_cols])
        cat_feature_names = ohe.get_feature_names_out(categorical_cols)
    else:
        X_cat_encoded = np.array([]).reshape(X.shape[0], 0)
        cat_feature_names = []
    
    # Combine with numerical features
    if numerical_cols:
        X_num = X[numerical_cols].values
        X_encoded = np.hstack([X_num, X_cat_encoded]) if X_cat_encoded.size > 0 else X_num
        feature_names = list(numerical_cols) + list(cat_feature_names)
    else:
        X_encoded = X_cat_encoded
        feature_names = list(cat_feature_names)
    
    print(f"Total features after encoding: {X_encoded.shape[1]}")
    
    return X_encoded, y_encoded, feature_names

def train_and_evaluate(model, model_name, X, y, cv_split):
    """
    Train a model and evaluate it.
    
    Parameters:
    -----------
    model : sklearn estimator
        Model to train
    model_name : str
        Name of the model
    X_encoded : np.ndarray
        features
    y_encoded : np.ndarray
        lables
    cv_split : cross validation splits
    
    Returns:
    --------
    dict : Dictionary with evaluation metrics
    """
    print(f"\nTraining {model_name}...")

    train_accs = []
    test_accs = []
    for i, (train_inds, test_inds) in enumerate(cv_split.split(X, y)):
        # Get the training and testing data for the current fold
        X_train, X_test = X[train_inds], X[test_inds]
        y_train, y_test = y[train_inds], y[test_inds]
        
        # Train
        model.fit(X_train, y_train)
    
        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_accs.append(accuracy_score(y_train, y_pred_train))
        test_accs.append(accuracy_score(y_test, y_pred_test))
    
    results = {
        'model_name': model_name,
        'train_accuracy_avg': np.mean(train_accs),
        'train_accuracy_std': np.std(train_accs),
        'test_accuracy_avg': np.mean(test_accs),
        'test_accuracy_std': np.std(test_accs),
        # 'y_pred_test': y_pred_test
    }
    
    print(f"  Train Accuracy: {results['train_accuracy_avg']:.4f}")
    print(f"  Test Accuracy: {results['test_accuracy_avg']:.4f}")
    
    return results

# special_params

def compare_models(datasets):
    """
    Compare XGBoost, DecisionTree, and RandomForest on multiple datasets.
    
    Parameters:
    -----------
    datasets : list of tuples
        List of (dataset_name, target_col) tuples
    """
    all_results = []
    
    for dataset_name in datasets:
        try:
            # Load dataset
            X, y = load_uci_dataset(dataset_name, None)
            if X is None or y is None:
                print(f"Skipping {dataset_name}...")
                continue
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            print(f"Skipping {dataset_name}...")
            continue

        # print(X.dtypes)
        # print(X.to_numpy().shape)
        # print(X[:5])
        # return

        n_cat = len(X.select_dtypes(include=['object', 'category']).columns.tolist())
        n_num = len(X.select_dtypes(include=[np.number]).columns.tolist())
    
        print(f"\nCategorical features: {n_cat}")
        print(f"Numerical features: {n_num}")
            
        if(n_num > n_cat):
            print(f"Skipping {dataset_name} because it has more numerical features than categorical features...")
            continue
        
        # Preprocess
        X_encoded, y_encoded, feature_names = preprocess_categorical(X, y)

        # 2. Configure KFold or use the default (cv=5 by default in recent versions)
        # cv_split = KFold(n_splits=5, shuffle=True, random_state=42)
        # cv_split = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_split = ShuffleSplit(test_size=.1, train_size=.9, n_splits=100, random_state=42)


        
        # # Split data
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        # )
        
        # print(f"\nTrain set: {X_train.shape[0]} samples")
        # print(f"Test set: {X_test.shape[0]} samples")
        
        # # Print label counts in training set
        # unique_labels, label_counts = np.unique(y_train, return_counts=True)
        # print(f"\nTraining set label counts:")
        # for label, count in zip(unique_labels, label_counts):
        #     print(f"  Label {label}: {count} samples ({count/len(y_train)*100:.2f}%)")
        

        # continue
        
        # Initialize models
        max_depth = 5
        models = {
            'DecisionTree': DecisionTreeClassifier(random_state=42, max_depth=max_depth),
            'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=max_depth, random_state=42, n_jobs=-1),
            'XGBoost': XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss', max_depth=max_depth),
            'STAND': STANDClassifier(**s_kwargs),
            'STAND (heir)': STANDClassifier(**s_kwargs, lam_p=lam_p, lam_e=lam_e, lam_l=lam_l)
        }
        
        # Train and evaluate each model
        dataset_results = []
        for model_name, model in models.items():
            # 3. Get cross-validation accuracies
            results = train_and_evaluate(model, model_name, X_encoded, y_encoded, cv_split)
            # print(model_name, results)


            # results = train_and_evaluate(
            #     model, model_name, X_train, X_test, y_train, y_test
            # )
            results['dataset'] = dataset_name
            dataset_results.append(results)
            all_results.append(results)
        
        # Print comparison for this dataset
        print(f"\n{'='*60}")
        print(f"Results Summary for {dataset_name}")
        print(f"{'='*60}")
        print(f"{'Model':<20} {'Train Acc':<15} {'Test Acc':<15}")
        print(f"{'-'*50}")
        for r in dataset_results:
            print(f"{r['model_name']:<20} {r['train_accuracy_avg']:<15.4f} {r['test_accuracy_avg']:<15.4f}")
        
        # Print detailed classification report for best model
        best_model_result = max(dataset_results, key=lambda x: x['test_accuracy_avg'])
        print(f"\nBest model: {best_model_result['model_name']} (Test Acc: {best_model_result['test_accuracy_avg']:.4f})")
        # print(f"\nClassification Report:")
        # print(classification_report(y_test, best_model_result['y_pred_test']))
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    summary_df = pd.DataFrame(all_results)
    summary_pivot = summary_df.pivot_table(
        values='test_accuracy_avg',
        index='dataset',
        columns='model_name',
        aggfunc='mean'
    )
    
    print("\nTest Accuracy by Dataset and Model:")
    pd.set_option('display.max_columns', None)
    print(summary_pivot)

    print("\nAverage Test Accuracy by Model:")
    avg_by_model = summary_df.groupby('model_name')['test_accuracy_avg'].mean()
    print(avg_by_model.sort_values(ascending=False))
    
    # Generate LaTeX table
    print("\n" + "="*60)
    print("LaTeX Table:")
    print("="*60)
    # Format the DataFrame for LaTeX with better column names
    latex_df = summary_pivot.copy()
    latex_df.index.name = 'Dataset'
    latex_table = latex_df.to_latex(
        float_format=lambda x: f'{x*100:.2f}\%',
        caption='Test Accuracy by Dataset and Model',
        label='tab:test_accuracy',
        escape=False,
        column_format='l' + 'c' * len(latex_df.columns)  # left-align first column, center others
    )
    print(latex_table)
    print("="*60)

    # Generate LaTeX table for average test accuracy by model
    print("\n" + "="*60)
    print("LaTeX Table: Average Test Accuracy by Model")
    print("="*60)
    avg_model_df = avg_by_model.sort_values(ascending=False).to_frame(name='Average Test Accuracy')
    avg_model_df.index.name = 'Model'
    latex_avg_table = avg_model_df.to_latex(
        float_format=lambda x: f'{x*100:.2f}\%',
        caption='Average Test Accuracy by Model',
        label='tab:avg_test_accuracy',
        escape=False,
        column_format='lc'
    )
    print(latex_avg_table)
    print("="*60)
    
    
    
    return all_results

# DESCRIPTION OF DATASETS
# https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame
# https://archive.ics.uci.edu/ml/datasets/congressional+voting+records
# http://archive.ics.uci.edu/ml/datasets/Breast+Cancer
# https://archive.ics.uci.edu/ml/datasets/hepatitis
# https://archive.ics.uci.edu/ml/datasets/zoo
# https://archive.ics.uci.edu/ml/datasets/Soybean+(Large)




if __name__ == "__main__":
    import openml

    all_multivariate_datasets = [    
        'hepatitis', 'breast-cancer', 'zoo', 'tic-tac-toe', 'vote', 
        #'heart-statlog', 'diabetes', 'ionosphere', 'sonar', 'wine', 'iris', 'glass', 
        'soybean'
        # 'lymphography', 'soybean', 'balance-scale',
        # 'hayes-roth', 'monk1', 'monk2', 'monk3', 'tae', 'cmc', 'flags',
        # 'dermatology', 'ecoli', 'yeast', 'abalone', 'page-blocks', 'shuttle',
        # 'letter', 'satimage', 'segment', 'waveform', 'pendigits', 'optdigits'
    ]
    
    # UCI datasets that are binary classification with categorical features
    # Using OpenML dataset names/IDs
    # datasets = [
        # ('adult', None),  # Adult income prediction
        # ('mushroom', None),  # Mushroom classification
        # ('credit-g', None),  # German credit
        # ('kr-vs-kp', None),  # King-Rook vs King-Pawn
        # ('splice', None),  # Splice junction gene sequences
    # ]
    
    print("="*60)
    print("UCI Binary Classification Dataset Comparison")
    print("Models: XGBoost, DecisionTree, RandomForest")
    print("="*60)
    
    results = compare_models(all_multivariate_datasets)
    
    print("\n" + "="*60)
    print("Comparison complete!")
    print("="*60)
