import argparse
import os
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from utils import ( get_sequence_alignment, 
                    save_best_model, 
                    save_best_parameters,
                    the_20_aa )
from one_hot_encoder import CustomizedOneHotEncoder

def run_nested_cross_validation(alignment, outer_folds, inner_folds, output_dir, bnAb, model_name, hyperparameters, ic_type, sensitivity_cutoff):
    best_params = {}
    for fold in range(outer_folds):
        fold += 1
        print(f'starting outer loop #{fold}')
        training_data = pd.read_csv(f'../training_test_data/{bnAb}/{bnAb}_fold{fold}_training_ic{ic_type}_{sensitivity_cutoff}cutoff.csv')
        training_data['sequence'] = np.array([alignment[row['virus_id']] for _, row in training_data.iterrows()])
        one_hot_encoder = CustomizedOneHotEncoder(categories=np.array(the_20_aa))
        training_data['sequence'] = one_hot_encoder.fit_transform(np.array(training_data['sequence']))

        #balance the training data by phenotype. No need to do this for validation data.
        pos_data = training_data[training_data['phenotype']==1]
        neg_data = training_data[training_data['phenotype']==0]
        if len(neg_data) > len(pos_data):
            add_data = pos_data.sample(n=len(neg_data)-len(pos_data), replace=True)
            pos_data = pd.concat([pos_data, add_data])
        if len(neg_data) < len(pos_data):
            add_data = neg_data.sample(n=len(pos_data)-len(neg_data), replace=True)
            neg_data = pd.concat([neg_data, add_data])
        training_data = pd.concat([pos_data, neg_data])
        training_data = training_data.sample(frac=1)#shuffle
        assert np.mean(training_data['phenotype']) == 0.5

        training_X = np.array(list(training_data['sequence']))
        training_Y = training_data['phenotype']
        training_pipeline = None
        if 'GBM' in model_name:
            gbm_classifier = GradientBoostingClassifier(n_iter_no_change=3, tol=0.001, loss='log_loss', criterion='friedman_mse', random_state=42)
            training_pipeline = Pipeline([
                (model_name, gbm_classifier)
            ])
        elif 'RF' in model_name:
            rf_classifier = RandomForestClassifier(random_state=42)
            training_pipeline = Pipeline([
                (model_name, rf_classifier)
            ])
    
        randomized_cv = RandomizedSearchCV( training_pipeline, 
                                            hyperparameters[model_name], 
                                            scoring='neg_log_loss',
                                            n_iter=10, 
                                            cv=inner_folds, 
                                            verbose=1, 
                                            random_state=42, 
                                            error_score='raise',
                                            n_jobs=-1)
        randomized_cv.fit(training_X, training_Y)
        #save best parameters
        best_params[f'fold{fold}'] = randomized_cv.best_params_
        #save best trained model
        save_best_model(randomized_cv.best_estimator_, output_dir, f'{model_name}_{bnAb}_fold{fold}_ic{ic_type}_{sensitivity_cutoff}cutoff')

    save_best_parameters(best_params, output_dir, f'{model_name}_{bnAb}_ic{ic_type}_{sensitivity_cutoff}cutoff')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', action='store', type=str, required=True, help='RF or GBM')
    parser.add_argument('-o', '--outer_folds', action='store', type=int, required=False, default=5)
    parser.add_argument('-i', '--inner_folds', action='store', type=int, required=False, default=5)
    parser.add_argument('-s', '--sensitivity_cutoff', action='store', type=int, required=True)
    parser.add_argument('-c', '--ic_type', action='store', type=int, required=True)
    parser.add_argument('-b', '--bnAb', action='store', type=str, required=True)
    args = parser.parse_args()

    output_dir = f'../final_trained_models'
    hyperparameters = {}
    if 'GBM' in args.model_name:
        hyperparameters[args.model_name] = {f'{args.model_name}__learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
                                            f'{args.model_name}__max_features': [0.03, 0.1, 0.2, 0.3, 0.5],
                                            f'{args.model_name}__max_depth': [1, 2, 3, 4, 5],
                                            f'{args.model_name}__n_estimators': [10, 50, 100, 500, 1000],
                                        }
    if 'RF' in args.model_name:
        hyperparameters[args.model_name] = { f'{args.model_name}__max_depth': [1, 2, 3, 4, 5], 
                                            f'{args.model_name}__max_features': [0.03, 0.1, 0.2, 0.3, 0.5],
                                            f'{args.model_name}__n_estimators': [10, 50, 100, 500, 1000],
                                        }
    alignment = get_sequence_alignment('../all_catnap_alignment.fasta')
    run_nested_cross_validation(alignment, args.outer_folds, args.inner_folds, output_dir, args.bnAb, args.model_name, hyperparameters, args.ic_type, args.sensitivity_cutoff)