import numpy as np
import pandas as pd
import os
import json
from utils import *

if __name__ == '__main__':
    model_name = 'ENS'
    for bnAb in bnAbs_of_interest+combos:
        print('bnAb', bnAb)
        bnAb_dir = f'../outputs/overall_performance/{bnAb}'
        if not os.path.isdir(bnAb_dir):
            os.mkdir(bnAb_dir)
        output_dir = os.path.join(bnAb_dir, model_name)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        
        performance = {}
        for fold in range(1,6,1):
            performance[f'fold{fold}'] = {}
            d_LBUM = pd.read_csv(f'../outputs/overall_performance/{bnAb}/LBUM/LBUM_fold{fold}_predictions.csv')
            d_RF = pd.read_csv(f'../outputs/overall_performance/{bnAb}/RF/RF_fold{fold}_predictions.csv')
            d_GBM = pd.read_csv(f'../outputs/overall_performance/{bnAb}/GBM/GBM_fold{fold}_predictions.csv')
            y_pred = []
            for _,r in d_LBUM.iterrows():
                v_id = r['virus_id']
                phenotype = r['phenotype']
                subtype = r['subtype']
                y_pred.append((v_id, subtype, phenotype, np.mean([r['predictions'], d_RF[d_RF['virus_id']==v_id]['predictions'].values[0], d_GBM[d_GBM['virus_id']==v_id]['predictions'].values[0]])))
            y_pred = pd.DataFrame(y_pred, columns=['virus_id', 'subtype', 'phenotype', 'predictions'])
            y_pred.to_csv(f'../outputs/overall_performance/{bnAb}/ENS/ENS_fold{fold}_predictions.csv')

            #save performance
            performance[f'fold{fold}'] = calculate_performance(y_pred, y_pred['predictions'].astype('float64'))
        performance_file_path = os.path.join(output_dir, f'{model_name}_performance.json')
        with open(performance_file_path, 'w') as pfp:
            json.dump(performance, pfp, indent=4)