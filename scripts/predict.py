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


def predict(base_params, one_hot_data, data, bnAb, n_folds, models, ic_type, cut_off):
    if 'best' in models:
        raise Exception('You cannot specify `best` with other models at the same time. When specifying `best`, it has to be the only argument.')
    if len(models) == 3 and 'ENS' not in models:
        models.append('ENS')
    for model_name in models:
        if model_name not in ['ENS', 'RF', 'GBM', 'LBUM']:
                raise Exception(f"{model_name} is not supported. Please choose from the following models: {['ENS','RF', 'GBM', 'LBUM']} or `best` ")
    output_data = {}
    models_to_run = ['ENS', 'RF', 'GBM', 'LBUM'] if 'ENS' in models else models
    for fold in range(n_folds):
        fold += 1
        print('fold', fold)
        with open(f'./hyperparameters/LBUM_fold{fold}_ic{ic_type}_{cut_off}cutoff_hyperparameters.json') as params_json:
            tuned_params = json.load(params_json)
        params = {  'pretrain_embedding_size':  base_params['pretrain_params']['embedding_size'], 
                    'pretrain_n_units': base_params['pretrain_params']['n_units'], 
                    'pretrain_n_layers': base_params['pretrain_params']['n_layers'], 
                    'pretrained_model_filepath': './pretrained_models/pretrained_model_epoch27.hdf5', 
                    'learning_rate': tuned_params['learning_rate'], 
                    'attention_neurons': tuned_params['attention_neurons'],
                    'n_layers_to_train':tuned_params['n_layers_to_train'],
                    'dropout_rate': tuned_params['dropout_rate'],
                    'classification_weight': tuned_params['classification_weight']
                }
        out_for_ens = {}
        for model_name in models_to_run:
            if model_name == 'LBUM':
                print('model', model_name)
                outs = {}
                for seed in [0, 10, 20, 30, 40]:
                    model = build_LBUM(params, dropout_on=True)
                    model.load_weights(f'./models/fold{fold}_{model_name}_{seed}seed_ic{ic_type}_{cut_off}cutoff.hdf5')
                    in_data = data.copy()
                    in_data['antibody_index'] = [all_antibodies.index(bnAb) for _ in range(len(in_data))]
                    X = form_language_model_input(in_data, inference=True)
                    all_predictions = None
                    all_predictions = np.stack([model.predict([X['left_input'], X['right_input'], X['antibody_index']], batch_size=32, verbose=1)[1] for _ in range(10)])
                    predictions = all_predictions.mean(axis=0)
                    for x,y in zip (predictions, data.iterrows()):
                        if y[1]['virus_id'] not in outs:
                            outs[y[1]['virus_id']] = []
                        outs[y[1]['virus_id']].append(x[0]) 
                for v_id in outs:
                    assert len(outs[v_id]) == 5
                    if v_id not in output_data: output_data[v_id] = {}
                    if model_name not in output_data[v_id]: output_data[v_id][model_name] = []
                    output_data[v_id][model_name].append(np.mean(outs[v_id]))
                    if v_id not in out_for_ens:
                        out_for_ens[v_id] = []
                    out_for_ens[v_id].append(np.mean(outs[v_id]))

            elif model_name in ['RF', 'GBM']:
                print('model', model_name)
                model = joblib.load(f'./models/{model_name}_{bnAb}_fold{fold}_ic{ic_type}_{cut_off}cutoff_best_model.pkl')
                predictions = None
                predictions = model.predict_proba(one_hot_data)
                predictions = predictions[:, 1]
                for x,y in zip (predictions, data.iterrows()):
                    v_id = y[1]['virus_id']
                    if v_id not in output_data: output_data[v_id] = {}
                    if model_name not in output_data[v_id]: output_data[v_id][model_name] = []
                    output_data[v_id][model_name].append(x)
                    if v_id not in out_for_ens:
                        out_for_ens[v_id] = []
                    out_for_ens[v_id].append(x)

        if 'ENS' in models_to_run:
            print('model', 'ENS')
            for v_id in out_for_ens:
                if len(out_for_ens[v_id]) != 3: print(len(out_for_ens[v_id]))
                assert len(out_for_ens[v_id]) == 3
                if 'ENS' not in output_data[v_id]:
                    output_data[v_id]['ENS'] = []
                output_data[v_id]['ENS'].append(np.mean(out_for_ens[v_id]))    

    data_to_write = []
    for virus_id in output_data:
        v = output_data[virus_id]
        row = [virus_id, ]
        for model in models:
            for i in range(n_folds):
                row.append(v[model][i])
        data_to_write.append(row)
    
    columns = ['virus_id']
    for m in models:
        for i in range(1,6,1):
            columns.append(f'{m} fold {i}')

    return data_to_write, columns

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--preprocessed_nonaligned_csv', action='store', type=str, required=True)
    parser.add_argument('-a', '--preprocessed_aligned_fasta', action='store', type=str, required=True)
    parser.add_argument('-o', '--output_dir', action='store', type=str, required=True)
    parser.add_argument('-p', '--prefix', action='store', type=str, required=True)
    parser.add_argument('-b', '--bnAb', action='store', type=str, required=True)
    parser.add_argument('-m', '--models', action='store', type=str, required=True, help=['best', 'RF', 'GBM', 'LBUM', 'ENS'])
    parser.add_argument('-f', '--config', type=str, default='./scripts/config.json', required=False)
    parser.add_argument('-i', '--ic', type=int, required=True, help='50 or 80')

    args = parser.parse_args()

    with open(args.config) as params_json:
        base_params = json.load(params_json)

    ic_type = args.ic
    cut_off = 50
    if ic_type == 80:
        cut_off = 1

    n_folds = 5
    data = pd.read_csv(args.preprocessed_nonaligned_csv)
    alignment = get_sequence_alignment(args.preprocessed_aligned_fasta)
    bnAb = args.bnAb
    if args.models.lower() == 'best':
        with open('./scripts/best_models.json') as models_json:
            best_models = json.load(models_json)
        if bnAb.lower() not in best_models[f'ic{ic_type}']:
            raise Exception(f'{bnAb} is not supported for the specified IC type. Please choose one of the following bnAbs: {list(best_models[f"ic{ic_type}"].keys())}')
        models = best_models[f'ic{ic_type}'][bnAb.lower()]
    else:
        models = args.models.split(',')

    one_hot_encoder = CustomizedOneHotEncoder(categories=np.array(the_20_aa))
    one_hot_data = one_hot_encoder.fit_transform(np.array([alignment[row['virus_id']] for _,row in data.iterrows()]))

    print('bnAb:', bnAb)
    print('models:', models)
    if bnAb not in bnAbs_of_interest:
        raise Exception(f'{bnAb} is not supported. Please choose one of the following bnAbs: {bnAbs_of_interest}')
    predictions, columns = predict(base_params, one_hot_data, data, bnAb, n_folds, models, ic_type, cut_off)
    pd.DataFrame(predictions, columns=columns).to_csv(os.path.join(args.output_dir, f'{bnAb}_{args.prefix}_ic{ic_type}_{cut_off}cutoff_resistance_probabilities.csv'))