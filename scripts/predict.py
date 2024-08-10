import numpy as np
import pandas as pd
import joblib, os, argparse, json
from utils import ( all_antibodies, 
                    get_sequence_alignment, 
                    the_20_aa,
                    form_language_model_input,
                    bnAbs_of_interest
                )
from rnn_models import build_LBUM
from one_hot_encoder import CustomizedOneHotEncoder


def predict(fine_tune_params, one_hot_data, data, bnAb, n_folds, models):
    output_data = {}
    for _,row in data.iterrows():
        output_data[row['virus_id']] = {}
        for model_name in models:
            output_data[row['virus_id']][model_name] = []
    for fold in range(n_folds):
        fold += 1
        print('fold', fold)
        params = {  'pretrain_embedding_size':  20, 
                    'pretrain_n_units': 512, 
                    'pretrain_n_layers': 2, 
                    'pretrained_model_filepath': './pretrained_models/pretrained_model_epoch27.hdf5', 
                    'learning_rate': fine_tune_params['learning_rate'], 
                    'attention_neurons': fine_tune_params['attention_neurons'],
                    'n_layers_to_train':fine_tune_params['n_layers_to_train'],
                    'dropout_rate': fine_tune_params['dropout_rate'],
                    'classification_weight': fine_tune_params['classification_weight']
                }
        for model_name in models:
            print('model', model_name)
            if model_name == 'LBUM':
                model = build_LBUM(params, dropout_on=True)
                model.load_weights(f'./final_trained_models/fold{fold}_{model_name}.hdf5')
                in_data = data.copy()
                in_data['antibody_index'] = [all_antibodies.index(bnAb) for _ in range(len(in_data))]
                X = form_language_model_input(in_data, inference=True)
                all_predictions = None
                all_predictions = np.stack([model.predict([X['left_input'], X['right_input'], X['antibody_index']], batch_size=32, verbose=1)[1] for _ in range(10)])
                predictions = all_predictions.mean(axis=0)
                for x,y in zip (predictions, data.iterrows()):
                    output_data[y[1]['virus_id']][model_name].append(x[0])
            else:
                model = joblib.load(f'./final_trained_models/{model_name}_{bnAb}_fold{fold}_best_model.pkl')
                predictions = None
                predictions = model.predict_proba(one_hot_data)
                predictions = predictions[:, 1]
                for x,y in zip (predictions, data.iterrows()):
                    output_data[y[1]['virus_id']][model_name].append(x)

    data_to_write = []
    for virus_id in output_data:
        v = output_data[virus_id]
        row = [virus_id, ]
        for model in models:
            for i in range(n_folds):
                row.append(v[model][i])
        data_to_write.append(row)
    return data_to_write

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--preprocessed_nonaligned_fasta', action='store', type=str, required=True)
    parser.add_argument('-o', '--output_dir', action='store', type=str, required=True)
    parser.add_argument('-p', '--prefix', action='store', type=str, required=True)
    parser.add_argument('-b', '--bnAbs', action='store', type=str, required=True)
    parser.add_argument('-m', '--models', action='store', type=str, required=True)
    parser.add_argument('-f', '--config', type=str, default='./scripts/config.json', required=False)

    args = parser.parse_args()

    with open(args.config) as params_json:
        params = json.load(params_json)
    fine_tune_params = params['fine_tune_params']

    n_folds = 5
    _, file_name = os.path.split(os.path.splitext(args.preprocessed_nonaligned_fasta)[0])
    data = pd.read_csv(os.path.join(args.output_dir, f'{file_name}.csv'))
    alignment = get_sequence_alignment(os.path.join(args.output_dir, f'aligned_{file_name}.fasta'))
    bnAbs = args.bnAbs.split(',')
    models = args.models.split(',')
    one_hot_encoder = CustomizedOneHotEncoder(categories=np.array(the_20_aa))
    one_hot_data = one_hot_encoder.fit_transform(np.array([alignment[row['virus_id']] for _,row in data.iterrows()]))

    
    columns = ['virus_id']
    for m in models:
        for i in range(1,6,1):
            columns.append(f'{m} fold {i}')
    for bnAb in bnAbs:
        print('bnAb', bnAb)
        if bnAb not in bnAbs_of_interest:
            raise Exception(f'{bnAb} is not supported. Please choose one of the following bnAbs: {bnAbs_of_interest}')
        predictions = predict(fine_tune_params, one_hot_data, data, bnAb, n_folds, models)
        pd.DataFrame(predictions, columns=columns).to_csv(os.path.join(args.output_dir, f'{bnAb}_{args.prefix}_resistance_probabilities.csv'))