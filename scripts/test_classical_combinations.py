import numpy as np
import pandas as pd
import joblib
import os
import json
from utils import ( get_sequence_alignment, 
                    calculate_regression_performance,
                    calculate_classification_performance,
                    save_predictions,
                    combos,
                    the_20_aa
                )
from one_hot_encoder import CustomizedOneHotEncoder

if __name__ == '__main__':
    for combo in combos:
        print('combo', combo)
        bnAb_dir = f'../outputs/overall_performance/{combo}'
        if not os.path.isdir(bnAb_dir):
            os.mkdir(bnAb_dir)
        testing_data = pd.read_csv(f'../datasets/training_test_data/{combo}/{combo}_all_phenotypes.csv')
        testing_data = testing_data[testing_data['right_censored'] == 0]
        testing_data['ic50'] = np.log10(testing_data['ic50'])
        alignment = get_sequence_alignment('../datasets/virseqs_aa_7_July_2022.fasta')
        one_hot_encoder = CustomizedOneHotEncoder(categories=np.array(the_20_aa))
        testing_data['sequence'] = np.array([alignment[row['virus_id']] for _, row in testing_data.iterrows()])
        testing_data['sequence'] = one_hot_encoder.fit_transform(np.array(testing_data['sequence']))
        for model_name in ['GBM_reg', 'RF_reg']:
            output_dir = os.path.join(bnAb_dir, model_name)
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            performance = {}
            for fold in range(5):
                fold += 1
                predictions = []
                for bnAb in combo.split('+'):
                    model = joblib.load(f'../final_trained_models/{bnAb}/{model_name}/{model_name}_{bnAb}_fold{fold}_best_model.pkl')
                    predictions.append(model.predict(np.array(list(testing_data['sequence']))))
                y_pred = None
                if len(predictions) == 2:
                    y_pred = np.array([np.log10(1/(1/(10**x) + 1/(10**y))) for x,y in zip(predictions[0], predictions[1])])
                else:
                    y_pred = np.array([np.log10(1/(1/(10**x) + 1/(10**y) + 1/(10**z))) for x,y,z in zip(predictions[0], predictions[1], predictions[2])])
                #save performance
                performance[f'fold{fold}'] = calculate_regression_performance(testing_data, y_pred.astype('float64'))
                save_predictions(testing_data, y_pred.astype('float64'), output_dir, f'{model_name.split("_")[0]}_fold{fold}_regression')
            performance_file_path = os.path.join(output_dir, f'{model_name.split("_")[0]}_regression_performance.json')
            with open(performance_file_path, 'w') as pfp:
                json.dump(performance, pfp, indent=4)

        for model_name in ['GBM_class', 'RF_class']:
            output_dir = os.path.join(bnAb_dir, model_name)
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            performance = {}
            for fold in range(5):
                fold += 1
                predictions = []
                for bnAb in combo.split('+'):
                    model = joblib.load(f'../final_trained_models/{bnAb}/{model_name}/{model_name}_{bnAb}_fold{fold}_best_model.pkl')
                    predictions.append(model.predict(np.array(list(testing_data['sequence']))))
                y_pred = None
                if len(predictions) == 2:
                    y_pred = np.array([x*y for x,y in zip(predictions[0], predictions[1])])
                else:
                    y_pred = np.array([x*y*z for x,y,z in zip(predictions[0], predictions[1], predictions[2])])
                #save performance
                performance[f'fold{fold}'] = calculate_classification_performance(testing_data, y_pred.astype('float64'))
                save_predictions(testing_data, y_pred.astype('float64'), output_dir, f'{model_name.split("_")[0]}_fold{fold}_classification')
            performance_file_path = os.path.join(output_dir, f'{model_name.split("_")[0]}_classification_performance.json')
            with open(performance_file_path, 'w') as pfp:
                json.dump(performance, pfp, indent=4)