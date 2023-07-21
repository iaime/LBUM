import numpy as np
import pandas as pd
import joblib, os, argparse
from utils import ( all_antibodies, 
                    get_sequence_alignment, 
                    the_20_aa,
                    form_language_model_input
                )
from rnn_models import build_LBUM
from one_hot_encoder import CustomizedOneHotEncoder


def predict(params, one_hot_data, data, bnAb_combination, n_folds, regression=False):
    models = ['GBM_class', 'RF_class', 'LBUM']
    if regression: 
        models = ['GBM_reg', 'RF_reg', 'LBUM']
    output_data = {}
    for _,row in data.iterrows():
        output_data[row['virus_id']] = {}
        for model_name in models:
            output_data[row['virus_id']][model_name] = []
    for fold in range(n_folds):
        fold += 1
        print('fold', fold)
        for model_name in models:
            print('model', model_name)
            if model_name == 'LBUM':
                model = build_LBUM(params, dropout_on=True)
                model.load_weights(f'../final_trained_models/fold{fold}_{model_name}.hdf5')
                all_y_pred = []
                for _ in range(10):
                    y_pred = []
                    for bnAb in bnAb_combination.split('+'):
                        in_data = data.copy()
                        in_data['bnAb'] = [all_antibodies.index(bnAb) for _ in range(len(in_data))]
                        X = form_language_model_input(in_data, inference_time=True)
                        if regression:
                            y_pred.append(model.predict([X['left_input'], X['right_input'], X['bnAb']], batch_size=32, verbose=1)[0])
                        else:
                            y_pred.append(model.predict([X['left_input'], X['right_input'], X['bnAb']], batch_size=32, verbose=1)[1])
                    if regression:
                        if len(y_pred) == 2:
                            all_y_pred.append(np.array([1/(1/(10**x) + 1/(10**y)) for x,y in zip(y_pred[0], y_pred[1])]))
                        else:
                            all_y_pred.append(np.array([1/(1/(10**x) + 1/(10**y) + 1/(10**z)) for x,y,z in zip(y_pred[0], y_pred[1], y_pred[2])]))
                    else:
                        if len(y_pred) == 2:
                            all_y_pred.append(np.array([x*y for x,y in zip(y_pred[0], y_pred[1])]))
                        else:
                            all_y_pred.append(np.array([x*y*z for x,y,z in zip(y_pred[0], y_pred[1], y_pred[2])]))
                        

                all_y_pred = np.stack(all_y_pred)
                predictions = all_y_pred.mean(axis=0)
                for x,y in zip (predictions, data.iterrows()):
                    output_data[y[1]['virus_id']][model_name].append(x[0])
            else:
                predictions = []
                for bnAb in bnAb_combination.split('+'):
                    model = joblib.load(f'../final_trained_models/{bnAb}/{model_name}/{model_name}_{bnAb}_fold{fold}_best_model.pkl')
                    if regression:
                        predictions.append(model.predict(one_hot_data))
                    else:
                        predictions.append(model.predict_proba(one_hot_data)[:, 1])
                if regression:
                    if len(predictions) == 2:
                        predictions = np.array([1/(1/(10**x) + 1/(10**y)) for x,y in zip(predictions[0], predictions[1])])
                    else:
                        predictions = np.array([1/(1/(10**x) + 1/(10**y) + 1/(10**z)) for x,y,z in zip(predictions[0], predictions[1], predictions[2])])
                else:
                    if len(predictions) == 2:
                        predictions = np.array([x*y for x,y in zip(predictions[0], predictions[1])])
                    else:
                        predictions = np.array([x*y*z for x,y,z in zip(predictions[0], predictions[1], predictions[2])])
                for x,y in zip (predictions, data.iterrows()):
                    output_data[y[1]['virus_id']][model_name].append(x)

    data_to_write = []
    for virus_id in output_data:
        v = output_data[virus_id]
        row = [virus_id, ]
        for model in models:
            for i in range(n_folds):
                if regression:
                    row.append(v[model][i]) 
                else:
                    row.append(v[model][i])
        data_to_write.append(row)
    return data_to_write

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--fasta_aligned_to_catnap', action='store', type=str, required=True)
    parser.add_argument('-d', '--data_csv', action='store', type=str, required=True)
    parser.add_argument('-o', '--output_dir', action='store', type=str, required=True)
    parser.add_argument('-p', '--prefix', action='store', type=str, required=True)
    parser.add_argument('-b', '--bnAbs_combinations', action='store', type=str, required=True)

    args = parser.parse_args()

    n_folds = 5
    data = pd.read_csv(args.data_csv)
    alignment = get_sequence_alignment(args.fasta_aligned_to_catnap)
    bnAbs_combinations = args.bnAbs_combinations.split(',')
    one_hot_encoder = CustomizedOneHotEncoder(categories=np.array(the_20_aa))
    one_hot_data = one_hot_encoder.fit_transform(np.array([alignment[row['virus_id']] for _,row in data.iterrows()]))

    params = {'pretrain_embedding_size':  20, 
            'pretrain_n_units': 512, 
            'pretrain_n_layers': 2, 
            'pretrained_model_filepath': '../pretrained_models/pretrained_model_epoch27.hdf5', 
            'learning_rate': 0.001, 
            'attention_neurons': 32,
            'n_layers_to_train': 100, 'dropout_rate': 0.4}
    columns = ['virus_id']
    for m in ['GBM', 'RF', 'LBUM']:
        for i in range(1,6,1):
            columns.append(f'{m} fold {i}')
    for combo in bnAbs_combinations:
        print('bnAb combination', combo)
        if len(combo.split('+')) > 3:
            raise Exception('So far we only support combinations of at most 3 bnAbs')
        regressions = predict(params, one_hot_data, data, combo, n_folds, regression=True)
        classifications = predict(params, one_hot_data, data, combo, n_folds, regression=False)
        pd.DataFrame(classifications, columns=columns).to_csv(os.path.join(args.output_dir, f'{combo}_{args.prefix}_classification_predictions.csv'))
        pd.DataFrame(regressions, columns=columns).to_csv(os.path.join(args.output_dir, f'{combo}_{args.prefix}_regression_predictions.csv'))
