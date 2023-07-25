import numpy as np
import pandas as pd
import joblib, os, argparse
import distutils.util
from utils import ( all_antibodies, 
                    get_sequence_alignment, 
                    the_20_aa,
                    form_language_model_input,
                    bnAbs_of_interest
                )
from rnn_models import build_LBUM
from one_hot_encoder import CustomizedOneHotEncoder


def predict(params, one_hot_data, data, bnAb, n_folds, model_names, regression=False):
    models = [x if x == 'LBUM' else f'{x}_class' for x in model_names]
    if regression: 
        models = [x if x == 'LBUM' else f'{x}_reg' for x in model_names]
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
                model.load_weights(f'./final_trained_models/fold{fold}_{model_name}.hdf5')
                in_data = data.copy()
                in_data['bnAb'] = [all_antibodies.index(bnAb) for _ in range(len(in_data))]
                X = form_language_model_input(in_data, inference_time=True)
                all_predictions = None
                if regression:
                    all_predictions = np.stack([model.predict([X['left_input'], X['right_input'], X['bnAb']], batch_size=32, verbose=1)[0] for _ in range(10)])
                else:
                    all_predictions = np.stack([model.predict([X['left_input'], X['right_input'], X['bnAb']], batch_size=32, verbose=1)[1] for _ in range(10)])
                predictions = all_predictions.mean(axis=0)
                for x,y in zip (predictions, data.iterrows()):
                    output_data[y[1]['virus_id']][model_name].append(x[0])
            else:
                model = joblib.load(f'./final_trained_models/{model_name}_{bnAb}_fold{fold}_best_model.pkl')
                predictions = None
                if regression:
                    predictions = model.predict(one_hot_data)
                else:
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
                if regression:
                    row.append(10**v[model][i]) 
                else:
                    row.append(v[model][i])
        data_to_write.append(row)
    return data_to_write

def predict_for_bnAb_combinations(params, one_hot_data, data, bnAb_combination, n_folds, model_names, regression=False):
    models = [x if x == 'LBUM' else f'{x}_class' for x in model_names]
    if regression: 
        models = [x if x == 'LBUM' else f'{x}_reg' for x in model_names]
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
                model.load_weights(f'./final_trained_models/fold{fold}_{model_name}.hdf5')
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
                    model = joblib.load(f'./final_trained_models/{model_name}_{bnAb}_fold{fold}_best_model.pkl')
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
    parser.add_argument('-a', '--preprocessed_nonaligned_fasta', action='store', type=str, required=True)
    parser.add_argument('-o', '--output_dir', action='store', type=str, required=True)
    parser.add_argument('-p', '--prefix', action='store', type=str, required=True)
    parser.add_argument('-b', '--bnAbs', action='store', type=str, required=True)
    parser.add_argument('-m', '--models', action='store', type=str, required=True)
    parser.add_argument('-r', '--regression', type=distutils.util.strtobool, required=True)
    parser.add_argument('-c', '--combination', type=distutils.util.strtobool, required=True)

    args = parser.parse_args()

    n_folds = 5
    _, file_name = os.path.split(os.path.splitext(args.preprocessed_nonaligned_fasta)[0])
    data = pd.read_csv(os.path.join(args.output_dir, f'{file_name}.csv'))
    alignment = get_sequence_alignment(os.path.join(args.output_dir, f'aligned_{file_name}.fasta'))
    bnAbs = args.bnAbs.split(',')
    models = args.models.split(',')
    one_hot_encoder = CustomizedOneHotEncoder(categories=np.array(the_20_aa))
    one_hot_data = one_hot_encoder.fit_transform(np.array([alignment[row['virus_id']] for _,row in data.iterrows()]))

    params = {'pretrain_embedding_size':  20, 
            'pretrain_n_units': 512, 
            'pretrain_n_layers': 2, 
            'pretrained_model_filepath': './pretrained_models/pretrained_model_epoch27.hdf5', 
            'learning_rate': 0.001, 
            'attention_neurons': 32,
            'n_layers_to_train': 100, 'dropout_rate': 0.4}
    columns = ['virus_id']
    for m in models:
        for i in range(1,6,1):
            columns.append(f'{m} fold {i}')
    for bnAb in bnAbs:
        print('bnAb', bnAb)
        if args.combination:
            if len(bnAb.split('+')) > 3 or len(bnAb.split('+')) < 2:
                raise Exception('So far we only support combinations of 2 or 3 bnAbs')
            for b in bnAb.split('+'):
                if b not in bnAbs_of_interest:
                    raise Exception(f'{b} is not supported. Please choose one of the following bnAbs: {bnAbs_of_interest}')
            predictions = predict_for_bnAb_combinations(params, one_hot_data, data, bnAb, n_folds, models, regression=args.regression)
            pd.DataFrame(predictions, columns=columns).to_csv(os.path.join(args.output_dir, f'{bnAb}_{args.prefix}_{"regression" if args.regression else "classification"}_predictions.csv'))
        else:
            if bnAb not in bnAbs_of_interest:
                raise Exception(f'{bnAb} is not supported. Please choose one of the following bnAbs: {bnAbs_of_interest}')
            predictions = predict(params, one_hot_data, data, bnAb, n_folds, models, regression=args.regression)
            pd.DataFrame(predictions, columns=columns).to_csv(os.path.join(args.output_dir, f'{bnAb}_{args.prefix}_{"regression" if args.regression else "classification"}_predictions.csv'))