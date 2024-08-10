import json
import pandas as pd
import numpy as np
import argparse
import os
import keras_tuner as kt
from pprint import pprint
from rnn_models import build_pretrain_model, sequence_vectorizer, CustomDropout
from sklearn.model_selection import train_test_split
from utils import form_language_model_input, all_antibodies, bnAbs_of_interest
import tensorflow as tf

def build_model(hp):
    pretrain_embedding_size = 20
    pretrain_n_units = 512 
    pretrain_n_layers = 2 
    pretrained_model_filepath = '../pretrained_models/pretrained_model_epoch27.hdf5'
    learning_rate = hp.Choice('learning_rate', values=[1e-4, 3e-4, 1e-3, 3e-3]) 
    attention_neurons = hp.Choice('attention_neurons', values=[32, 64, 128, 256])
    dropout_rate = hp.Choice('dropout_rate', values=[0.1, 0.2, 0.3, 0.4, 0.5])
    n_layers_to_train = hp.Choice('n_layers_to_train', values=[17, 19, 21, 23])
    classification_weight = hp.Choice('classification_weight', values=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    model = build_pretrain_model(   embedding_size=pretrain_embedding_size,
                                    n_units=pretrain_n_units,
                                    n_layers=pretrain_n_layers,
                                )
    print('loading pretrained model')
    model.load_weights(pretrained_model_filepath)
    
    left_input = tf.keras.layers.Input(shape=(1,), name='left_input', dtype='string')
    right_input = tf.keras.layers.Input(shape=(1,), name='right_input', dtype='string')
    left_X = sequence_vectorizer(left_input)
    right_X = sequence_vectorizer(right_input)

    #retrieve pretrained_model's layers
    left_X = model.get_layer('input_embeddings')(left_X)
    right_X = model.get_layer('input_embeddings')(right_X)
    for i in range(pretrain_n_layers):
        left_X = model.get_layer(f'left_lstm_{i}')(left_X)
        right_X = model.get_layer(f'right_lstm_{i}')(right_X)
    right_X = model.get_layer('backward_embeds_reverser')(right_X)
    concatenated_embeddings = tf.keras.layers.concatenate([left_X, right_X], name='embedding_layer')
    
    #retrieve context for input bnAb
    bnAb_input = tf.keras.layers.Input(shape=(1,), name='bnAb_input', dtype='int64')
    bnAbs_contexts = tf.keras.layers.Embedding(    input_dim=len(all_antibodies), 
                                                output_dim=attention_neurons, 
                                                mask_zero=False,
                                                embeddings_initializer='glorot_uniform',
                                                name='bnAbs_contexts'
                                            )
    
    context = bnAbs_contexts(bnAb_input)

    concatenated_embeddings = CustomDropout(dropout_rate)(concatenated_embeddings)
    attention = tf.keras.layers.Dense(attention_neurons, activation='tanh', name='attention_network')(concatenated_embeddings)
    attention = tf.keras.layers.dot([attention, context], -1)
    attention = tf.keras.layers.Flatten()(attention)
    attention = tf.keras.layers.Activation('softmax', name='attention_layer')(attention)
    mod_attention = tf.keras.layers.RepeatVector(pretrain_n_units*2)(attention)
    mod_attention = tf.keras.layers.Permute([2, 1])(mod_attention)
    output = tf.keras.layers.Multiply()([concatenated_embeddings, mod_attention])
    output = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1), name='attentive_embedding')(output)
    output = CustomDropout(dropout_rate)(output)
    regression_output = tf.keras.layers.Dense(1, name='regression_output')(output)
    classification_output = tf.keras.layers.Dense(1, activation=None, name='logits_output')(output)
    classification_output = tf.keras.layers.Activation('sigmoid', name='classification_output')(classification_output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    fine_tune_model = tf.keras.models.Model([left_input, right_input, bnAb_input], [regression_output, classification_output])
    #Let's freeze lower layers' weights to avoid wrecking them
    for layer in fine_tune_model.layers[:-n_layers_to_train]:
        layer.trainable = False
    fine_tune_model.compile(loss=['mean_squared_error', 'binary_crossentropy'], loss_weights=[1 - classification_weight, classification_weight], optimizer=optimizer, weighted_metrics={'regression_output':['mae'], 'classification_output':['AUC', 'accuracy']})
    print('------fine-tuning model summary------')
    print(fine_tune_model.summary(show_trainable=True))
    return fine_tune_model

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sensitivity_cutoff', action='store', type=int, required=True)
    parser.add_argument('-c', '--ic_type', action='store', type=int, required=True)
    parser.add_argument('-o', '--fold', action='store', type=str, required=True)
    parser.add_argument('-r', '--seed', action='store', type=int, required=True)
    args = parser.parse_args()

    #set seed
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()

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

    training_X = form_language_model_input(training_data)
    val_X = form_language_model_input(validation_data)
    val_Y_class = validation_data['phenotype']
    training_Y_class = training_data['phenotype']
    val_Y_reg = np.log10(validation_data[f'ic{args.ic_type}'].astype(float))
    training_Y_reg = np.log10(training_data[f'ic{args.ic_type}'].astype(float))
    
    tuner = kt.BayesianOptimization(
                            build_model,
                            objective=kt.Objective('val_classification_output_loss', direction='min'),
                            max_trials=10,
                            seed=1,
                            overwrite=False,
                            # distribution_strategy=tf.distribute.MirroredStrategy(),
                            project_name=f'../LBUM_{args.seed}_hypertuning_fold{args.fold}_ic{args.ic_type}_{args.sensitivity_cutoff}cutoff'
                        )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)

    tuner.search(   [training_X['left_input'], training_X['right_input'], training_X['antibody_index']], [training_Y_reg, training_Y_class],
                    validation_data=([val_X['left_input'], val_X['right_input'], val_X['antibody_index']], [val_Y_reg, val_Y_class], [val_X['regression_weight'], val_X['classification_weight']]),
                    sample_weight={'regression_output': training_X['regression_weight'], 'classification_output': training_X['classification_weight']},
                    epochs=100,
                    batch_size=32,
                    use_multiprocessing=False,
                    verbose=1,
                    callbacks = [early_stopping]
                )
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    pprint(best_hyperparameters.values)
    file_name = os.path.join('../hyperparameters', f'LBUM_fold{args.fold}_ic{args.ic_type}_{args.sensitivity_cutoff}cutoff_hyperparameters.json')
    with open(file_name, 'w') as pfp:
        json.dump(best_hyperparameters.values, pfp, indent=4)
