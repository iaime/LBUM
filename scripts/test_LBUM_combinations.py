import numpy as np
import pandas as pd
import argparse
import json
import os
from rnn_models import *
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', action='store', type=str, required=False, default='LBUM')
    parser.add_argument('-c', '--combo', action='store', type=str, required=True)
    parser.add_argument('-p', '--config', action='store', type=str, required=True)

    args = parser.parse_args()

    with open(args.config) as params_json:
        params = json.load(params_json)
    pretrain_params = params['pretrain_params']
    fine_tune_params = params['fine_tune_params']

    combo = args.combo
    data = pd.read_csv(f'../datasets/training_test_data/{combo}/{combo}_all_phenotypes.csv')

    print('combo', combo)
    bnAb_dir = f'../outputs/overall_performance/{combo}'
    if not os.path.isdir(bnAb_dir):
        os.mkdir(bnAb_dir)
    model_name = args.model_name
    output_dir = os.path.join(bnAb_dir, model_name)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    performance = {}
    for fold in range(5):
        fold += 1
        testing_data = data.copy()
        testing_data['ic50'] = np.log10(testing_data['ic50'])
        testing_data = testing_data[testing_data['right_censored']==0]
        LBUM_model = build_LBUM({'pretrain_embedding_size':  pretrain_params['embedding_size'], 
                                    'pretrain_n_units': pretrain_params['n_units'], 
                                    'pretrain_n_layers': pretrain_params['n_layers'], 
                                    'pretrained_model_filepath': '../pretrained_models/pretrained_model_epoch27.hdf5', 
                                    'learning_rate': fine_tune_params['learning_rate'], 
                                    'attention_neurons': fine_tune_params['attention_neurons'],
                                    'n_layers_to_train':fine_tune_params['n_layers_to_train'],
                                    'dropout_rate': fine_tune_params['dropout_rate']}, dropout_on=True)
        LBUM_model.load_weights(f'../final_trained_models/fold{fold}_{model_name}.hdf5')
        all_y_pred = []
        for i in range(10):
            predictions = []
            for bnAb in combo.split('+'):
                LBUM_testing_data = testing_data.copy()
                LBUM_testing_data['bnAb'] = [all_antibodies.index(bnAb) for _ in range(len(LBUM_testing_data))]
                LBUM_testing_data['regression_weight'] = [1 for _ in range(len(LBUM_testing_data))]
                LBUM_testing_data['classification_weight'] = [1 for _ in range(len(LBUM_testing_data))]
                LBUM_testing_data = form_language_model_input(LBUM_testing_data)
                predictions.append(LBUM_model.predict([LBUM_testing_data['left_input'], LBUM_testing_data['right_input'], LBUM_testing_data['bnAb']], batch_size=32, verbose=1)[0])
            if len(predictions) == 2:
                all_y_pred.append(np.array([np.log10(1/(1/(10**x) + 1/(10**y))) for x,y in zip(predictions[0], predictions[1])]))
            else:
                all_y_pred.append(np.array([np.log10(1/(1/(10**x) + 1/(10**y) + 1/(10**z))) for x,y,z in zip(predictions[0], predictions[1], predictions[2])]))
        all_y_pred = np.stack(all_y_pred)
        y_pred = all_y_pred.mean(axis=0)
        y_std = all_y_pred.std(axis=0)

        #save performance
        performance[f'fold{fold}'] = calculate_regression_performance(testing_data, y_pred.astype('float64'))
        save_predictions(testing_data, y_pred.astype('float64'), output_dir, f'{model_name}_fold{fold}_regression', std=y_std)
    performance_file_path = os.path.join(output_dir, f'{model_name}_regression_performance.json')
    with open(performance_file_path, 'w') as pfp:
        json.dump(performance, pfp, indent=4)

    # #classification case
    performance = {}
    for fold in range(5):
        fold += 1
        testing_data = data.copy()
        LBUM_model = build_LBUM({'pretrain_embedding_size':  pretrain_params['embedding_size'], 
                                    'pretrain_n_units': pretrain_params['n_units'], 
                                    'pretrain_n_layers': pretrain_params['n_layers'], 
                                    'pretrained_model_filepath': '../pretrained_models/pretrained_model_epoch27.hdf5', 
                                    'learning_rate': fine_tune_params['learning_rate'], 
                                    'attention_neurons': fine_tune_params['attention_neurons'],
                                    'n_layers_to_train':fine_tune_params['n_layers_to_train'],
                                    'dropout_rate': fine_tune_params['dropout_rate']}, dropout_on=True)
        LBUM_model.load_weights(f'../final_trained_models/fold{fold}_{model_name}.hdf5')
        all_y_pred = []
        for i in range(10):
            predictions = []
            for bnAb in combo.split('+'):
                LBUM_testing_data = testing_data.copy()
                LBUM_testing_data['bnAb'] = [all_antibodies.index(bnAb) for _ in range(len(LBUM_testing_data))]
                LBUM_testing_data['regression_weight'] = [1 for _ in range(len(LBUM_testing_data))]
                LBUM_testing_data['classification_weight'] = [1 for _ in range(len(LBUM_testing_data))]
                LBUM_testing_data = form_language_model_input(LBUM_testing_data)
                predictions.append(LBUM_model.predict([LBUM_testing_data['left_input'], LBUM_testing_data['right_input'], LBUM_testing_data['bnAb']], batch_size=32, verbose=1)[1])
            if len(predictions) == 2:
                all_y_pred.append(np.array([x*y for x,y in zip(predictions[0], predictions[1])]))
            else:
                all_y_pred.append(np.array([x*y*z for x,y,z in zip(predictions[0], predictions[1], predictions[2])]))
        all_y_pred = np.stack(all_y_pred)
        y_pred = all_y_pred.mean(axis=0)
        y_std = all_y_pred.std(axis=0)

        #save performance
        performance[f'fold{fold}'] = calculate_classification_performance(testing_data, y_pred.astype('float64'))
        save_predictions(testing_data, y_pred.astype('float64'), output_dir, f'{model_name}_fold{fold}_classification', std=y_std)
    performance_file_path = os.path.join(output_dir, f'{model_name}_classification_performance.json')
    with open(performance_file_path, 'w') as pfp:
        json.dump(performance, pfp, indent=4)