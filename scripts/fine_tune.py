import tensorflow as tf
tf.keras.utils.set_random_seed(0)
tf.config.experimental.enable_op_determinism()
import argparse
import json
import os, sys
import pandas as pd
import numpy as np
from rnn_models import build_LBUM, SaveLossesAndMetrics
from sklearn.model_selection import train_test_split
from utils import get_today_date_dir, form_language_model_input, all_antibodies, bnAbs_of_interest

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--pretrained_model_filepath', action='store', type=str, required=True)
    parser.add_argument('-p', '--config', action='store', type=str, required=True)
    parser.add_argument('-f', '--fine_tuning_model_name', action='store', type=str, required=True)
    parser.add_argument('-o', '--fold', action='store', type=int, required=True)

    args = parser.parse_args()

    with open(args.config) as params_json:
        params = json.load(params_json)

    today_output_dir = get_today_date_dir(params['output_dir'])
    pretrain_params = params['pretrain_params']
    fine_tune_params = params['fine_tune_params']
    
    all_training_data = []
    for i, bnAb in enumerate(all_antibodies):
        bnAb_dir = os.path.join('../datasets/training_test_data', bnAb)
        training_data = None
        if bnAb not in bnAbs_of_interest:
            training_data = pd.read_csv(os.path.join(bnAb_dir, f'{bnAb}_training.csv'))
        else:
            training_data = pd.read_csv(os.path.join(bnAb_dir, f'{bnAb}_fold{args.fold}_training.csv'))
        training_data['bnAb'] = [i for _ in range(len(training_data))]
        training_data['regression_weight'] = [0 if x['right_censored'] == 1 else 1 for _,x in training_data.iterrows()]
        training_data['classification_weight'] = [1 for _,x in training_data.iterrows()]
        all_training_data.append(training_data)  
    all_training_data = pd.concat(all_training_data)

    training_data, validation_data = train_test_split(all_training_data, stratify=all_training_data['phenotype'], test_size=0.1, random_state=1)
    training_X = form_language_model_input(training_data)
    validation_X = form_language_model_input(validation_data)
    validation_Y_reg = np.log10(validation_data['ic50'])
    validation_Y_class = validation_data['phenotype']
    training_Y_reg = np.log10(training_data['ic50'])
    training_Y_class = training_data['phenotype']

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
    model_filepath = os.path.join(today_output_dir, 
        f'fold{args.fold}_{args.fine_tuning_model_name}.hdf5')

    model_check_point = tf.keras.callbacks.ModelCheckpoint(
        model_filepath,
        save_best_only=True, save_weights_only=False,
        mode='auto', save_freq='epoch',
    )
    losses_and_metrics_filepath = os.path.join( today_output_dir, 
        f'fold{args.fold}_{args.fine_tuning_model_name}_losses_and_metrics.csv')
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
                                        'dropout_rate': fine_tune_params['dropout_rate']})

    
    print('..................fitting the model....................')
    fine_tuning_model.fit(  [training_X['left_input'], training_X['right_input'], training_X['bnAb']], [training_Y_reg, training_Y_class],
                            validation_data=([validation_X['left_input'], validation_X['right_input'], validation_X['bnAb']], [validation_Y_reg, validation_Y_class], [validation_X['regression_weight'], validation_X['classification_weight']]),
                            sample_weight={'regression_output': training_X['regression_weight'], 'classification_output': training_X['classification_weight']},
                            batch_size=fine_tune_params['batch_size'],
                            epochs=fine_tune_params['epochs'], 
                            callbacks=callbacks,
                            use_multiprocessing=False,
                            verbose=fine_tune_params['verbose']
                        )    