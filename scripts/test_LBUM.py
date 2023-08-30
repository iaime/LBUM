import numpy as np
import pandas as pd
import argparse
import os
import json
from rnn_models import *
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', action='store', type=str, required=False, default='LBUM')
    parser.add_argument('-b', '--bnAb', action='store', type=str, required=True)
    parser.add_argument('-p', '--config', action='store', type=str, required=True)

    args = parser.parse_args()

    with open(args.config) as params_json:
        params = json.load(params_json)
    pretrain_params = params['pretrain_params']
    fine_tune_params = params['fine_tune_params']

    bnAb = args.bnAb
    print('bnAb', bnAb)
    bnAb_dir = f'../outputs/overall_performance/{bnAb}'
    if not os.path.isdir(bnAb_dir):
        os.mkdir(bnAb_dir)
    model_name = args.model_name
    output_dir = os.path.join(bnAb_dir, model_name)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    performance = {}
    for fold in range(1,6,1):
        fold = str(fold)
        print('fold', fold)
        testing_data = pd.read_csv(f'../training_test_data/{bnAb}/{bnAb}_fold{fold}_testing.csv')
        LBUM_testing_data = testing_data.copy()
        LBUM_testing_data['antibody_index'] = [all_antibodies.index(bnAb) for _ in range(len(LBUM_testing_data))]
        LBUM_testing_data = form_language_model_input(LBUM_testing_data, inference=True)
        LBUM_model = build_LBUM(  {     'pretrain_embedding_size':  pretrain_params['embedding_size'], 
                                        'pretrain_n_units': pretrain_params['n_units'], 
                                        'pretrain_n_layers': pretrain_params['n_layers'], 
                                        'pretrained_model_filepath': '../pretrained_models/pretrained_model_epoch27.hdf5', 
                                        'learning_rate': fine_tune_params['learning_rate'], 
                                        'attention_neurons': fine_tune_params['attention_neurons'],
                                        'n_layers_to_train':fine_tune_params['n_layers_to_train'],
                                        'dropout_rate': fine_tune_params['dropout_rate'],
                                        'classification_weight': fine_tune_params['classification_weight']}, 
                                        dropout_on=True)
        LBUM_model.load_weights(f'../final_trained_models/fold{fold}_{model_name}.hdf5')

        all_y_pred = np.stack([LBUM_model.predict([LBUM_testing_data['left_input'], LBUM_testing_data['right_input'], LBUM_testing_data['antibody_index']], batch_size=32, verbose=1)[1] for _ in range(10)])
        y_pred = all_y_pred.mean(axis=0)
        y_std = all_y_pred.std(axis=0)

        #save performance
        performance[f'fold{fold}'] = calculate_performance(testing_data, y_pred.astype('float64'))
        test_data_copy = pd.DataFrame.copy(testing_data)
        test_data_copy = test_data_copy.drop('sequence', axis=1)
        save_predictions(test_data_copy, y_pred.astype('float64'), output_dir, f'{model_name}_fold{fold}', std=y_std)
    performance_file_path = os.path.join(output_dir, f'{model_name}_performance.json')
    with open(performance_file_path, 'w') as pfp:
        json.dump(performance, pfp, indent=4)