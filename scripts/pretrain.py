import argparse
import numpy as np
import pandas as pd
import json
import os
import tensorflow as tf
tf.keras.utils.set_random_seed(0)
tf.config.experimental.enable_op_determinism()
from rnn_models import build_pretrain_model, SaveLossesAndMetrics, sequence_vectorizer
from utils import get_today_date_dir, form_language_model_input
from sklearn.model_selection import train_test_split

def pretrain( model, training_X, training_Y, validation_X, validation_Y, model_filename, batch_size=32, workers=1, 
                epochs=3, verbose=1, output_dir='./'):
    model_filepath = os.path.join(output_dir, model_filename + '_epoch{epoch:02d}.hdf5')
    model_check_point = tf.keras.callbacks.ModelCheckpoint(
        model_filepath,
        save_best_only=False, save_weights_only=False,
        mode='auto', save_freq='epoch',
    )
    losses_and_metrics_filepath = os.path.join(output_dir, f'{model_filename}_pretraining_losses_and_metrics.csv')  
    logs_callback = SaveLossesAndMetrics(losses_and_metrics_filepath)
    callbacks = [model_check_point, logs_callback]

    model.fit(  training_X, training_Y,
                validation_data=(validation_X, validation_Y),
                batch_size=batch_size,
                use_multiprocessing=True,
                workers=workers,
                epochs=epochs,
                verbose=verbose,
                callbacks=callbacks)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--config', action='store', type=str, required=True)
    parser.add_argument('-m', '--pretrain_model_name', action='store', type=str, required=True)
    parser.add_argument('-f', '--pretraining_data_filepath', action='store', type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as params_json:
        params = json.load(params_json)

    today_output_dir = get_today_date_dir(params['output_dir'])
    pretraining_data = pd.read_pickle(args.pretraining_data_filepath)
    
    pretraining_data.index.name = 'virus_id'
    pretraining_data = pretraining_data.reset_index()
    pretraining_data = pretraining_data.sample(frac=1)#shuffle input data
    training_data, validation_data = train_test_split(pretraining_data, test_size=0.1, random_state=1)

    print('number of training sequences:', len(training_data))
    print('number of validation sequences:', len(validation_data))
    model = build_pretrain_model()

    training_X = form_language_model_input(training_data, fine_tuning=False)
    validation_X = form_language_model_input(validation_data, fine_tuning=False)
    training_Y = sequence_vectorizer(training_data['sequence'])
    validation_Y = sequence_vectorizer(validation_data['sequence'])

    
    pretraining_params = params['pretrain_params']
    pretrain(   model, 
                [training_X['left_input'], training_X['right_input']], training_Y, 
                [validation_X['left_input'], validation_X['right_input']], validation_Y, 
                args.pretrain_model_name, 
                batch_size=pretraining_params['batch_size'], 
                workers=pretraining_params['workers'], 
                epochs=pretraining_params['epochs'], 
                verbose=pretraining_params['verbose'], 
                output_dir=today_output_dir)
