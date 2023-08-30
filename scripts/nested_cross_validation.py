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
from sklearn.calibration import CalibratedClassifierCV


def run_nested_cross_validation(alignment, outer_folds, inner_folds, output_dir, bnAb, model_name, hyperparameters):
    best_params = {}
    for fold in range(outer_folds):
        fold += 1
        print(f'starting outer loop #{fold}')
        training_data = pd.read_csv(f'../training_test_data/{bnAb}/{bnAb}_fold{fold}_training.csv')
        training_data['sequence'] = np.array([alignment[row['virus_id']] for _, row in training_data.iterrows()])
        one_hot_encoder = CustomizedOneHotEncoder(categories=np.array(the_20_aa))
        training_data['sequence'] = one_hot_encoder.fit_transform(np.array(training_data['sequence']))
        training_X = np.array(list(training_data['sequence']))
        training_Y = training_data['phenotype']
        training_pipeline = None
        if 'GBM' in model_name:
            gbm_classifier = GradientBoostingClassifier(n_iter_no_change=3, tol=0.001, loss='log_loss', criterion='friedman_mse', random_state=42)
            training_pipeline = Pipeline([
                (model_name, CalibratedClassifierCV(base_estimator=gbm_classifier, cv=5))
            ])
        elif 'RF' in model_name:
            rf_classifier = RandomForestClassifier(random_state=42)
            training_pipeline = Pipeline([
                (model_name, CalibratedClassifierCV(base_estimator=rf_classifier, cv=5))
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
        save_best_model(randomized_cv.best_estimator_, output_dir, f'{model_name}_{bnAb}_fold{fold}')

    save_best_parameters(best_params, output_dir, f'{model_name}_{bnAb}')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', action='store', type=str, required=True, help='RF or GBM')
    parser.add_argument('-o', '--outer_folds', action='store', type=int, required=False, default=5)
    parser.add_argument('-i', '--inner_folds', action='store', type=int, required=False, default=5)
    parser.add_argument('-b', '--bnAb', action='store', type=str, required=True)
    args = parser.parse_args()

    output_dir = f'../final_trained_models'
    hyperparameters = {}
    if 'GBM' in args.model_name:
        hyperparameters[args.model_name] = {f'{args.model_name}__base_estimator__learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
                                            f'{args.model_name}__base_estimator__max_features': [0.03, 0.1, 0.2, 0.3, 0.5],
                                            f'{args.model_name}__base_estimator__max_depth': [1, 2, 3, 4, 5],
                                            f'{args.model_name}__base_estimator__n_estimators': [10, 50, 100, 500, 1000],
                                            f'{args.model_name}__method': ['isotonic', 'sigmoid']
                                        }
    if 'RF' in args.model_name:
        hyperparameters[args.model_name] = { f'{args.model_name}__base_estimator__max_depth': [1, 2, 3, 4, 5], 
                                            f'{args.model_name}__base_estimator__max_features': [0.03, 0.1, 0.2, 0.3, 0.5],
                                            f'{args.model_name}__base_estimator__n_estimators': [10, 50, 100, 500, 1000],
                                            f'{args.model_name}__method': ['isotonic', 'sigmoid']
                                        }
    alignment = get_sequence_alignment('../all_catnap_alignment.fasta')
    run_nested_cross_validation(alignment, args.outer_folds, args.inner_folds, output_dir, args.bnAb, args.model_name, hyperparameters)