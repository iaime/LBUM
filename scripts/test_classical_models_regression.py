import numpy as np
import pandas as pd
import joblib
import os
import json
from utils import ( get_sequence_alignment, 
                    calculate_regression_performance,
                    save_predictions,
                    bnAbs_of_interest,
                    the_20_aa
                )
from pprint import pprint
from one_hot_encoder import CustomizedOneHotEncoder

if __name__ == '__main__':
    for bnAb in bnAbs_of_interest:
        print('bnAb', bnAb)
        bnAb_dir = f'../outputs/overall_performance/{bnAb}'
        if not os.path.isdir(bnAb_dir):
            os.mkdir(bnAb_dir)
        for model_name in ['GBM_reg', 'RF_reg']:
            output_dir = os.path.join(bnAb_dir, model_name)
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            performance = {}
            for fold in range(5):
                fold += 1
                testing_data = pd.read_csv(f'../datasets/training_test_data/{bnAb}/{bnAb}_fold{fold}_testing.csv')
                testing_data = testing_data[testing_data['right_censored'] == 0]
                testing_data['ic50'] = np.log10(testing_data['ic50'])
                model = joblib.load(f'../final_trained_models/{bnAb}/{model_name}/{model_name}_{bnAb}_fold{fold}_best_model.pkl')
                alignment = get_sequence_alignment('../datasets/virseqs_aa_7_July_2022.fasta')
                one_hot_encoder = CustomizedOneHotEncoder(categories=np.array(the_20_aa))

                testing_data['sequence'] = np.array([alignment[row['virus_id']] for _, row in testing_data.iterrows()])
                testing_data['sequence'] = one_hot_encoder.fit_transform(np.array(testing_data['sequence']))
                
                y_pred = model.predict(np.array(list(testing_data['sequence'])))

                #save performance
                performance[f'fold{fold}'] = calculate_regression_performance(testing_data, y_pred.astype('float64'))
                save_predictions(testing_data, y_pred.astype('float64'), output_dir, f'{model_name.split("_")[0]}_fold{fold}_regression')

            performance_file_path = os.path.join(output_dir, f'{model_name.split("_")[0]}_regression_performance.json')
            with open(performance_file_path, 'w') as pfp:
                json.dump(performance, pfp, indent=4)