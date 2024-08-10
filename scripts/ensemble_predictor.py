import numpy as np
import pandas as pd
import os
import json
import argparse
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sensitivity_cutoff', action='store', type=int, required=True)
    parser.add_argument('-c', '--ic_type', action='store', type=int, required=True)
    args = parser.parse_args()

    model_name = 'ENS'
    prefix = 'outside'
    for bnAb in bnAbs_of_interest:
        file_name = f'../performance/{bnAb}/GBM/{prefix}_GBM_fold1_ic{args.ic_type}_{args.sensitivity_cutoff}cutoff_predictions.csv'
        if not os.path.isfile(file_name): continue
        print('bnAb', bnAb)
        bnAb_dir = f'../performance/{bnAb}'
        if not os.path.isdir(bnAb_dir):
            os.mkdir(bnAb_dir)
        output_dir = os.path.join(bnAb_dir, model_name)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        performance = {}
        for fold in range(1,6,1):
            performance[f'fold{fold}'] = {}
            
            d_LBUM = pd.read_csv(f'../performance/{bnAb}/LBUM/{prefix}_LBUM_fold{fold}_ic{args.ic_type}_{args.sensitivity_cutoff}cutoff_predictions.csv')
            d_RF = pd.read_csv(f'../performance/{bnAb}/RF/{prefix}_RF_fold{fold}_ic{args.ic_type}_{args.sensitivity_cutoff}cutoff_predictions.csv')
            d_GBM = pd.read_csv(f'../performance/{bnAb}/GBM/{prefix}_GBM_fold{fold}_ic{args.ic_type}_{args.sensitivity_cutoff}cutoff_predictions.csv')
            y_pred = []
            for _,r in d_LBUM.iterrows():
                v_id = r['virus_id']
                phenotype = r['phenotype']
                # subtype = r['subtype']
                y_pred.append((v_id, phenotype, np.mean([r['predictions'], d_RF[d_RF['virus_id']==v_id]['predictions'].values[0], d_GBM[d_GBM['virus_id']==v_id]['predictions'].values[0]])))
            y_pred = pd.DataFrame(y_pred, columns=['virus_id', 'phenotype', 'predictions'])
            y_pred.to_csv(f'../performance/{bnAb}/ENS/{prefix}_ENS_fold{fold}_ic{args.ic_type}_{args.sensitivity_cutoff}cutoff_predictions.csv')

            #save performance
            performance[f'fold{fold}'] = calculate_performance(y_pred, y_pred['predictions'].astype('float64'))
        performance_file_path = os.path.join(output_dir, f'{prefix}_{model_name}_ic{args.ic_type}_{args.sensitivity_cutoff}cutoff_performance.json')
        with open(performance_file_path, 'w') as pfp:
            json.dump(performance, pfp, indent=4)