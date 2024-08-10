import argparse
import json
import os, sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from rnn_models import build_LBUM, SaveLossesAndMetrics
from utils import form_language_model_input, all_antibodies, bnAbs_of_interest

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--pretrained_model_filepath', action='store', type=str, required=True)
    parser.add_argument('-p', '--config', action='store', type=str, required=True)
    parser.add_argument('-f', '--fine_tuning_model_name', action='store', type=str, required=True)
    parser.add_argument('-o', '--fold', action='store', type=str, required=True)
    parser.add_argument('-r', '--seed', action='store', type=int, required=True)
    parser.add_argument('-s', '--sensitivity_cutoff', action='store', type=int, required=True)
    parser.add_argument('-c', '--ic_type', action='store', type=int, required=True)

    args = parser.parse_args()

    #set seed
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()

    with open(args.config) as params_json:
        base_params = json.load(params_json)

    with open(f'../hyperparameters/{args.fine_tuning_model_name}_fold{args.fold}_ic{args.ic_type}_{args.sensitivity_cutoff}cutoff_hyperparameters.json') as params_json:
        tuned_params = json.load(params_json)
    
    print('finetuning hyperparameters:')
    print(tuned_params)

    trained_model_dir = '../final_trained_models'
    
    all_training_data = []
    all_validation_data = []
    number_of_training_abs = 0
    number_of_training_bnAbs = 0
    for i, bnAb in enumerate(all_antibodies):
        missing_data_message = f'>>>>> skipping {bnAb} because only one phenotype is available in the training data'
        training_data_dir = f'../training_test_data/{bnAb}'
        training_data = None
        if bnAb not in bnAbs_of_interest:
            file_name = os.path.join(training_data_dir, f'{bnAb}_training_ic{args.ic_type}_{args.sensitivity_cutoff}cutoff.csv')
            if not os.path.isfile(file_name):
                print(missing_data_message)
                continue
            training_data = pd.read_csv(file_name)
        else:
            file_name = os.path.join(training_data_dir, f'{bnAb}_fold{args.fold}_training_ic{args.ic_type}_{args.sensitivity_cutoff}cutoff.csv')
            if not os.path.isfile(file_name):
                print(missing_data_message)
                continue
            training_data = pd.read_csv(file_name)
            #we only validate on bnAbs of interest (i.e., testing bnAbs)
            training_data, validation_data = train_test_split(training_data, stratify=training_data['phenotype'], test_size=0.1, random_state=1)
            validation_data['antibody_index'] = [i for _ in range(len(validation_data))]
            all_validation_data.append(validation_data)
        training_data['antibody_index'] = [i for _ in range(len(training_data))]

        #balance the training data by phenotype. No need to do this for validation data.
        pos_data = training_data[training_data['phenotype']==1]
        neg_data = training_data[training_data['phenotype']==0]
        if len(neg_data) == 0 or len(pos_data)==0:
            print(missing_data_message)
            continue
        if len(neg_data) > len(pos_data):
            add_data = pos_data.sample(n=len(neg_data)-len(pos_data), replace=True)
            pos_data = pd.concat([pos_data, add_data])
        if len(neg_data) < len(pos_data):
            add_data = neg_data.sample(n=len(pos_data)-len(neg_data), replace=True)
            neg_data = pd.concat([neg_data, add_data])
        training_data = pd.concat([pos_data, neg_data])
        assert np.mean(training_data['phenotype']) == 0.5
        all_training_data.append(training_data)
        if bnAb in bnAbs_of_interest:
            number_of_training_bnAbs += 1
        else:
            number_of_training_abs += 1
    print('effective total of training abs:', number_of_training_abs)
    print('effective total of training bnAbs:', number_of_training_bnAbs)
    training_data = pd.concat(all_training_data)
    training_data['regression_weight'] = [0 if x['right_censored'] == 1 else 1 for _,x in training_data.iterrows()]#for regularization
    training_data['classification_weight'] = [1 for _,x in training_data.iterrows()]
    training_data = training_data.sample(frac=1)#shuffle training data

    validation_data = pd.concat(all_validation_data)
    validation_data['regression_weight'] = [0 for _,x in validation_data.iterrows()]#we only care about classification performance to stop training a model
    validation_data['classification_weight'] = [1 for _,x in validation_data.iterrows()]
 
    print('training data size:', len(training_data), 'validation data size:', len(validation_data))
    training_X = form_language_model_input(training_data)
    validation_X = form_language_model_input(validation_data)
    validation_Y_class = validation_data['phenotype']
    training_Y_class = training_data['phenotype']
    validation_Y_reg = np.log10(validation_data[f'ic{args.ic_type}'].astype(float))
    training_Y_reg = np.log10(training_data[f'ic{args.ic_type}'].astype(float))

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
    model_filepath = os.path.join(trained_model_dir, f'fold{args.fold}_{args.fine_tuning_model_name}_{args.seed}seed_ic{args.ic_type}_{args.sensitivity_cutoff}cutoff.hdf5')

    model_check_point = tf.keras.callbacks.ModelCheckpoint(
        model_filepath,
        save_best_only=True, save_weights_only=False,
        mode='auto', save_freq='epoch',
    )
    losses_and_metrics_filepath = os.path.join( trained_model_dir, f'fold{args.fold}_{args.fine_tuning_model_name}_{args.seed}seed_ic{args.ic_type}_{args.sensitivity_cutoff}cutoff_losses_and_metrics.csv')
    logs_callback = SaveLossesAndMetrics(losses_and_metrics_filepath)
    callbacks = None
    fine_tuning_model = None

    callbacks = [early_stopping, logs_callback, model_check_point]
    fine_tuning_model = build_LBUM({   'pretrain_embedding_size':  base_params['pretrain_params']['embedding_size'], 
                                        'pretrain_n_units': base_params['pretrain_params']['n_units'], 
                                        'pretrain_n_layers': base_params['pretrain_params']['n_layers'], 
                                        'pretrained_model_filepath': args.pretrained_model_filepath, 
                                        'learning_rate': tuned_params['learning_rate'], 
                                        'attention_neurons': tuned_params['attention_neurons'],
                                        'n_layers_to_train':tuned_params['n_layers_to_train'],
                                        'dropout_rate': tuned_params['dropout_rate'],
                                        'classification_weight': tuned_params['classification_weight']})

    
    print('..................fitting the model....................')
    fine_tuning_model.fit(  [training_X['left_input'], training_X['right_input'], training_X['antibody_index']], [training_Y_reg, training_Y_class],
                            validation_data=([validation_X['left_input'], validation_X['right_input'], validation_X['antibody_index']], [validation_Y_reg, validation_Y_class], [validation_X['regression_weight'], validation_X['classification_weight']]),
                            sample_weight={'regression_output': training_X['regression_weight'], 'classification_output': training_X['classification_weight']},
                            batch_size=base_params['fine_tune_params']['batch_size'],
                            epochs=base_params['fine_tune_params']['epochs'], 
                            callbacks=callbacks,
                            use_multiprocessing=False,
                            verbose=base_params['fine_tune_params']['verbose']
                        )    