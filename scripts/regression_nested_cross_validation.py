import argparse
import os
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from utils import ( get_sequence_alignment, 
                    save_best_model, 
                    save_best_parameters,
                    the_20_aa )
from one_hot_encoder import CustomizedOneHotEncoder


def run_nested_cross_validation(alignment, outer_folds, inner_folds, output_dir, bnAb, model_name, hyperparameters):
    best_params = {}
    for fold in range(outer_folds):
        fold += 1
        print(f'starting outer loop #{fold}')
        training_data = pd.read_csv(f'../datasets/training_test_data/{bnAb}/{bnAb}_fold{fold}_training.csv')
        training_data = training_data[training_data['right_censored'] == 0]
        training_data['sequence'] = np.array([alignment[row['virus_id']] for virus_id, row in training_data.iterrows()])
        one_hot_encoder = CustomizedOneHotEncoder(categories=np.array(the_20_aa))
        training_data['sequence'] = one_hot_encoder.fit_transform(np.array(training_data['sequence']))
        training_X = np.array(list(training_data['sequence']))
        training_Y = np.log10(training_data['ic50'])
        training_pipeline = None
        if 'GBM' in model_name:
            gbm_classifier = GradientBoostingRegressor(n_iter_no_change=3, tol=0.001, loss='squared_error', criterion='friedman_mse', random_state=42)
            training_pipeline = Pipeline([
                (model_name, gbm_classifier)
            ])
        elif 'RF' in model_name:
            rf_classifier = RandomForestRegressor(random_state=42)
            training_pipeline = Pipeline([
                (model_name, rf_classifier)
            ])
    
        randomized_cv = RandomizedSearchCV( training_pipeline, 
                                            hyperparameters[model_name], 
                                            scoring='neg_mean_squared_error',
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

    save_best_parameters(best_params, output_dir, model_name)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--config', action='store', type=str, required=True)
    parser.add_argument('-m', '--model_name', action='store', type=str, required=True)
    parser.add_argument('-o', '--outer_folds', action='store', type=int, required=False, default=5)
    parser.add_argument('-i', '--inner_folds', action='store', type=int, required=False, default=5)
    parser.add_argument('-b', '--bnAb', action='store', type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as params_json:
        params = json.load(params_json)

    bnAb_dir = f'../final_trained_models/{args.bnAb}'
    output_dir = os.path.join(bnAb_dir, args.model_name)
    if not os.path.isdir(bnAb_dir):
        os.mkdir(bnAb_dir)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    hyperparameters = {}
    if 'GBM' in args.model_name:
        hyperparameters[args.model_name] = {f'{args.model_name}__learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
                                            f'{args.model_name}__max_features': [0.03, 0.1, 0.2, 0.3, 0.5],
                                            f'{args.model_name}__max_depth': [1, 2, 3, 4, 5, 6],
                                            f'{args.model_name}__n_estimators': [10, 50, 100, 500, 1000]
                                        }
    if 'RF' in args.model_name:
        hyperparameters[args.model_name] = { f'{args.model_name}__max_depth': [1, 2, 3, 4, 5, 6], 
                                            f'{args.model_name}__max_features': [0.03, 0.1, 0.2, 0.3, 0.5],
                                            f'{args.model_name}__n_estimators': [10, 50, 100, 500, 1000]
        }
    alignment = get_sequence_alignment(params['raw_data']['catnap_fasta'])
    run_nested_cross_validation(alignment, args.outer_folds, args.inner_folds, output_dir, args.bnAb, args.model_name, hyperparameters)