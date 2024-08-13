import pandas as pd
import numpy as np
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from utils import the_20_aa, all_antibodies
# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

protein_vocabulary = ['B', 'Z']
protein_vocabulary.extend(the_20_aa)
maximum_sequence_length = 900

def standardizer(tensor_str):
    #makes all amino acids upper case.
    #assumes that non-amino acids characters have been removed
    return tf.strings.upper(tensor_str)

def split_by_character(tensor_str):
    return tf.strings.bytes_split(tensor_str)

sequence_vectorizer = tf.keras.layers.TextVectorization(
    output_mode='int',
    standardize=standardizer,
    split=split_by_character,
    vocabulary=protein_vocabulary,
    output_sequence_length=maximum_sequence_length,
    name='sequence_vectorizer'
)

def build_pretrain_model(   embedding_size=20, 
                            n_units=512, 
                            n_layers=2, 
                            learning_rate=0.001):
    embedding_layer = tf.keras.layers.Embedding(input_dim=len(protein_vocabulary) + 2,# we add 2 because OOV and mask (i.e. 1 and 0 ) tokens 
                                                output_dim=embedding_size, 
                                                mask_zero=True,
                                                name='input_embeddings'
                                            )

    left_input = tf.keras.layers.Input(shape=(1,), name='left_input', dtype='string')
    right_input = tf.keras.layers.Input(shape=(1,), name='right_input', dtype='string')
    left_X = sequence_vectorizer(left_input)
    right_X = sequence_vectorizer(right_input)
    left_X = embedding_layer(left_X)
    right_X = embedding_layer(right_X)
    
    for i in range(n_layers):
        left_X = tf.keras.layers.LSTM(n_units, return_sequences=True, name=f'left_lstm_{i}')(left_X)
        right_X = tf.keras.layers.LSTM(n_units, go_backwards=True, return_sequences=True, name=f'right_lstm_{i}')(right_X)

    #IMPORTANT: we have to reverse the sequences of hidden states of the backward lstm before concatenation.
    #Otherwise, the model cheats
    right_X = tf.keras.layers.Lambda(lambda x: tf.keras.backend.reverse(x, 1), name='backward_embeds_reverser') (right_X)
    X = tf.keras.layers.concatenate([left_X, right_X], name='embedding_layer')
    output = tf.keras.layers.Dense(len(protein_vocabulary) + 2, name='output_layer', activation='softmax')(X)
    model = tf.keras.models.Model([left_input, right_input], output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(   optimizer=optimizer, 
                            loss='sparse_categorical_crossentropy', 
                            metrics=['accuracy']
                        )
    # print('------pretraining model summary------')
    # print(model.summary())
    return model

def build_LBUM(params, dropout_on=True):
    pretrain_embedding_size = params['pretrain_embedding_size'] 
    pretrain_n_units = params['pretrain_n_units'] 
    pretrain_n_layers = params['pretrain_n_layers'] 
    pretrained_model_filepath = params['pretrained_model_filepath']
    learning_rate = params['learning_rate'] 
    attention_neurons = params['attention_neurons']
    dropout_rate = params['dropout_rate']
    n_layers_to_train = params['n_layers_to_train']
    classification_weight = params['classification_weight']

    model = build_pretrain_model(   embedding_size=pretrain_embedding_size,
                                    n_units=pretrain_n_units,
                                    n_layers=pretrain_n_layers,
                                )
    # print('loading pretrained model')
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

    #include multi-context attention
    if dropout_on:
        concatenated_embeddings = CustomDropout(dropout_rate)(concatenated_embeddings)
    else:
        concatenated_embeddings = DropoutOff(dropout_rate)(concatenated_embeddings)
    attention = tf.keras.layers.Dense(attention_neurons, activation='tanh', name='attention_network')(concatenated_embeddings)
    attention = tf.keras.layers.dot([attention, context], -1)
    attention = tf.keras.layers.Flatten()(attention)
    attention = tf.keras.layers.Activation('softmax', name='attention_layer')(attention)
    mod_attention = tf.keras.layers.RepeatVector(pretrain_n_units*2)(attention)
    mod_attention = tf.keras.layers.Permute([2, 1])(mod_attention)
    output = tf.keras.layers.Multiply()([concatenated_embeddings, mod_attention])
    output = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1), name='attentive_embedding')(output)
    if dropout_on:
        output = CustomDropout(dropout_rate)(output)
    else:
        output = DropoutOff(dropout_rate)(output)
    regression_output = tf.keras.layers.Dense(1, name='regression_output')(output)
    classification_output = tf.keras.layers.Dense(1, activation=None, name='logits_output')(output)
    classification_output = tf.keras.layers.Activation('sigmoid', name='classification_output')(classification_output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    fine_tune_model = tf.keras.models.Model([left_input, right_input, bnAb_input], [regression_output, classification_output])
    #Let's freeze lower layers' weights to avoid wrecking them
    for layer in fine_tune_model.layers[:-n_layers_to_train]:
        layer.trainable = False
    fine_tune_model.compile(loss=['mean_squared_error', 'binary_crossentropy'], loss_weights=[1 - classification_weight, classification_weight], optimizer=optimizer, weighted_metrics={'regression_output':['mae'], 'classification_output':['AUC']})
    # print('------fine-tuning model summary------')
    # print(fine_tune_model.summary(show_trainable=True))
    return fine_tune_model

class CustomDropout(tf.keras.layers.Dropout):
    def call(self, X):
        return super().call(X, training=True)

class DropoutOff(tf.keras.layers.Dropout):
    def call(self, X):
        return super().call(X, training=False)

class SaveLossesAndMetrics(tf.keras.callbacks.Callback):
    def __init__(self, losses_and_metrics_filepath):
        self.logs_filepath = losses_and_metrics_filepath
        self.loss_and_metrics = {}
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
    
    def on_epoch_end(self, epoch, logs=None):
        self.loss_and_metrics[epoch] = logs

    def on_train_end(self, logs=None):
        pd.DataFrame.from_dict(self.loss_and_metrics, orient='index').to_csv(self.logs_filepath, index_label='epochs')