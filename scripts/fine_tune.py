import argparse
import json
import os, sys
import pandas as pd
import numpy as np
import tensorflow as tf
tf.keras.utils.set_random_seed(0)
tf.config.experimental.enable_op_determinism()
from sklearn.model_selection import train_test_split
from rnn_models import build_LBUM, SaveLossesAndMetrics
from utils import form_language_model_input, all_antibodies, bnAbs_of_interest

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--pretrained_model_filepath', action='store', type=str, required=True)
    parser.add_argument('-p', '--config', action='store', type=str, required=True)
    parser.add_argument('-f', '--fine_tuning_model_name', action='store', type=str, required=True)
    parser.add_argument('-o', '--fold', action='store', type=str, required=True)

    args = parser.parse_args()

    with open(args.config) as params_json:
        params = json.load(params_json)

    trained_model_dir = '../final_trained_models'
    pretrain_params = params['pretrain_params']
    fine_tune_params = params['fine_tune_params']
    
    all_training_data = []
    for i, bnAb in enumerate(all_antibodies):
        training_data_dir = f'../training_test_data/{bnAb}'
        training_data = None
        if bnAb not in bnAbs_of_interest:
            training_data = pd.read_csv(os.path.join(training_data_dir, f'{bnAb}_training.csv'))
        else:
            training_data = pd.read_csv(os.path.join(training_data_dir, f'{bnAb}_fold{args.fold}_training.csv'))
        training_data['antibody_index'] = [i for _ in range(len(training_data))]
        all_training_data.append(training_data)  
    all_training_data = pd.concat(all_training_data)
    all_training_data['regression_weight'] = [0 if x['right_censored'] == 1 else 1 for _,x in all_training_data.iterrows()]
    all_training_data['classification_weight'] = [1 for _,x in all_training_data.iterrows()]
    
    training_data, validation_data = train_test_split(all_training_data, stratify=all_training_data['phenotype'], test_size=0.1, random_state=1)
    print('training data size:', len(training_data), 'validation data size:', len(validation_data))
    training_X = form_language_model_input(training_data)
    validation_X = form_language_model_input(validation_data)
    validation_Y_class = validation_data['phenotype']
    training_Y_class = training_data['phenotype']
    validation_Y_reg = np.log10(validation_data['ic50'].astype(float))
    training_Y_reg = np.log10(training_data['ic50'].astype(float))

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
    model_filepath = os.path.join(trained_model_dir, f'fold{args.fold}_{args.fine_tuning_model_name}.hdf5')

    model_check_point = tf.keras.callbacks.ModelCheckpoint(
        model_filepath,
        save_best_only=True, save_weights_only=False,
        mode='auto', save_freq='epoch',
    )
    losses_and_metrics_filepath = os.path.join( trained_model_dir, f'fold{args.fold}_{args.fine_tuning_model_name}_losses_and_metrics.csv')
    logs_callback = SaveLossesAndMetrics(losses_and_metrics_filepath)
    callbacks = None
    fine_tuning_model = None

    callbacks = [early_stopping, logs_callback, model_check_point]
    fine_tuning_model = build_LBUM({   'pretrain_embedding_size':  pretrain_params['embedding_size'], 
                                        'pretrain_n_units': pretrain_params['n_units'], 
                                        'pretrain_n_layers': pretrain_params['n_layers'], 
                                        'pretrained_model_filepath': args.pretrained_model_filepath, 
                                        'learning_rate': fine_tune_params['learning_rate'], 
                                        'attention_neurons': fine_tune_params['attention_neurons'],
                                        'n_layers_to_train':fine_tune_params['n_layers_to_train'],
                                        'dropout_rate': fine_tune_params['dropout_rate'],
                                        'classification_weight': fine_tune_params['classification_weight']})

    
    print('..................fitting the model....................')
    fine_tuning_model.fit(  [training_X['left_input'], training_X['right_input'], training_X['antibody_index']], [training_Y_reg, training_Y_class],
                            validation_data=([validation_X['left_input'], validation_X['right_input'], validation_X['antibody_index']], [validation_Y_reg, validation_Y_class], [validation_X['regression_weight'], validation_X['classification_weight']]),
                            sample_weight={'regression_output': training_X['regression_weight'], 'classification_output': training_X['classification_weight']},
                            batch_size=fine_tune_params['batch_size'],
                            epochs=fine_tune_params['epochs'], 
                            callbacks=callbacks,
                            use_multiprocessing=False,
                            verbose=fine_tune_params['verbose']
                        )    