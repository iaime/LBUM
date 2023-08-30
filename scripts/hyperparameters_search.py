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
tf.keras.utils.set_random_seed(0)
tf.config.experimental.enable_op_determinism()

def build_model(hp):
    pretrain_embedding_size = 20
    pretrain_n_units = 512 
    pretrain_n_layers = 2 
    pretrained_model_filepath = '../pretrained_models/pretrained_model_epoch27.hdf5'
    learning_rate = hp.Choice('learning_rate', values=[1e-4, 3e-4, 1e-3, 3e-3]) 
    attention_neurons = hp.Choice('attention_neurons', values=[32, 64, 128, 256])
    dropout_rate = hp.Choice('dropout_rate', values=[0.1, 0.2, 0.3, 0.4, 0.5])
    n_layers_to_train = hp.Choice('n_layers_to_train', values=[20, 22, 24, 26])
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
    
    bnAb_temperatures = tf.keras.layers.Embedding(  input_dim=len(all_antibodies), 
                                                    output_dim=1, 
                                                    mask_zero=False,
                                                    embeddings_initializer=tf.keras.initializers.Constant(1.5),
                                                    name='bnAbs_temperatures'
                                                )
    
    context = bnAbs_contexts(bnAb_input)
    temperature = bnAb_temperatures(bnAb_input)

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
    temperature = tf.keras.layers.Flatten()(temperature)
    classification_output = tf.keras.layers.Lambda(lambda x: x[0]/x[1])([classification_output, temperature])
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

    all_training_data = []
    for i, nAb in enumerate(all_antibodies):
        nAb_dir = os.path.join('../training_test_data', nAb)
        training_data = None
        if nAb in bnAbs_of_interest: 
            continue
        else:
            training_data = pd.read_csv(os.path.join(nAb_dir, f'{nAb}_training.csv'))
        training_data['antibody_index'] = [i for _ in range(len(training_data))]
        all_training_data.append(training_data)
    all_training_data = pd.concat(all_training_data)

    training_data, val_data = train_test_split(all_training_data, stratify=all_training_data['phenotype'], test_size=0.1, random_state=1)

    training_data['regression_weight'] = [0 if x['right_censored'] == 1 else 1 for _,x in training_data.iterrows()]
    training_data['classification_weight'] = [1 for _,x in training_data.iterrows()]
    val_data['regression_weight'] = [0 if x['right_censored'] == 1 else 1 for _,x in val_data.iterrows()]
    val_data['classification_weight'] = [1 for _,x in val_data.iterrows()]

    training_X = form_language_model_input(training_data)
    val_X = form_language_model_input(val_data)
    val_Y_class = val_data['phenotype']
    training_Y_class = training_data['phenotype']
    val_Y_reg = np.log10(val_data['ic50'].astype(float))
    training_Y_reg = np.log10(training_data['ic50'].astype(float))
    
    tuner = kt.BayesianOptimization(
                            build_model,
                            objective=kt.Objective('val_classification_output_loss', direction='min'),
                            max_trials=10,
                            seed=1,
                            overwrite=True,
                            project_name=f'LBUM_hypertuning'
                        )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

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
