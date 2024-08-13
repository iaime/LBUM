import os, csv, time
import pandas as pd
import numpy as np
import joblib
from Bio import SeqIO
from sklearn.metrics import roc_auc_score, roc_curve, log_loss, accuracy_score, precision_recall_curve, auc, matthews_corrcoef, precision_score
from imblearn.metrics import  sensitivity_score, specificity_score
from sklearn.model_selection import StratifiedKFold

the_20_aa = [   'A',
                'R',
                'N',
                'D',
                'C',
                'Q',
                'E',
                'G',
                'H',
                'I',
                'L',
                'K',
                'M',
                'F',
                'P',
                'S',
                'T',
                'W',
                'Y',
                'V',
            ]

hxb2 = '''MRVKEKYQHLWRWGWRWGTMLLGMLMICSATEKLWVTVYYGVPVWKEATTTLFCASDAKAYDTEVHNVWATHACVPTDPNPQEVVLVNVTENFNMWKNDMVEQMHEDIISLWDQSLKPCVKLTPLCVSLKCTDLKNDTNTNSSSGRMIMEKGEIKNCSFNISTSIRGKVQKEYAFFYKLDIIPIDNDTTSYKLTSCNTSVITQACPKVSFEPIPIHYCAPAGFAILKCNNKTFNGTGPCTNVSTVQCTHGIRPVVSTQLLLNGSLAEEEVVIRSVNFTDNAKTIIVQLNTSVEINCTRPNNNTRKRIRIQRGPGRAFVTIGKIGNMRQAHCNISRAKWNNTLKQIASKLREQFGNNKTIIFKQSSGGDPEIVTHSFNCGGEFFYCNSTQLFNSTWFNSTWSTEGSNNTEGSDTITLPCRIKQIINMWQKVGKAMYAPPISGQIRCSSNITGLLLTRDGGNSNNESEIFRPGGGDMRDNWRSELYKYKVVKIEPLGVAPTKAKRRVVQREKRAVGIGALFLGFLGAAGSTMGAASMTLTVQARQLLSGIVQQQNNLLRAIEAQQHLLQLTVWGIKQLQARILAVERYLKDQQLLGIWGCSGKLICTTAVPWNASWSNKSLEQIWNHTTWMEWDREINNYTSLIHSLIEESQNQQEKNEQELLELDKWASLWNWFNITNWLWYIKLFIMIVGGLVGLRIVFAVLSIVNRVRQGYSPLSFQTHLPTPRGPDRPEGIEEEGGERDRDRSIRLVNGSLALIWDDLRSLCLFSYHRLRDLLLIVTRIVELLGRRGWEALKYWWNLLQYWSQELKNSAVSLLNATAIAVAEGTDRVIEVVQGACRAIRHIPRRIRQGLERILL'''

#these env coordinates are 1-indexed 
epitope_coordinates = {
    'C3_V3': (296, 354),
    'CD4bs': (124, 477),
    'MPER': (662, 683),
    'gp120_gp41': (495, 618),
    'V1_V2': (131, 166)
}

grouped_bnAbs_of_interest = {
    'C3_V3': ['2G12', 'PGT128', 'PGT121', '10-1074', 'PGT135', 'DH270.1', 'DH270.5', 'DH270.6', 'VRC29.03'],
    'CD4bs': ['3BNC117', 'b12', 'VRC01', 'VRC07', 'HJ16', 'NIH45-46', 'VRC-CH31', 'VRC-PG04', 'VRC03', 'VRC13'],
    'MPER': ['2F5', '4E10'],
    'gp120_gp41': ['8ANC195', '35O22', 'PGT151', 'VRC34.01'],
    'V1_V2': ['PG9', 'PG16', 'CH01', 'PGT145', 'VRC26.25', 'PGDM1400', 'VRC38.01', 'VRC26.08']
}

bnAbs_to_epitope = {}
for epi in grouped_bnAbs_of_interest:
    for bnAb in grouped_bnAbs_of_interest[epi]:
        bnAbs_to_epitope[bnAb] = epi

all_epitopes = ['C3_V3', 'CD4bs', 'MPER', 'gp120_gp41', 'V1_V2']

bnAbs_of_interest = [   '2F5',
                        '2G12',
                        '3BNC117',
                        '4E10',
                        '8ANC195',
                        '10-1074',
                        '35O22',
                        'b12',
                        'CH01',
                        'DH270.1',
                        'DH270.5',
                        'DH270.6',
                        'HJ16',
                        'NIH45-46',
                        'PG9',
                        'PG16',
                        'PGDM1400',
                        'PGT121',
                        'PGT128',
                        'PGT135',
                        'PGT145',
                        'PGT151',
                        'VRC-CH31',
                        'VRC-PG04',
                        'VRC01',
                        'VRC03',
                        'VRC07',
                        'VRC13',
                        'VRC26.08',
                        'VRC26.25',
                        'VRC29.03',
                        'VRC34.01',
                        'VRC38.01'
                    ]

all_antibodies = [  '697-D',
                    'PCT64-35G',
                    'b6',
                    'VRC34.06',
                    'PGT130',
                    '8ANC134',
                    'PCDN-38A',
                    '1357',
                    'PGDM21',
                    'A12',
                    'VRC34.04',
                    'VRC26.22',
                    'JM4-IgG2b',
                    'BG18',
                    'VRC38.08',
                    '10A3',
                    'VRC24',
                    'DH270.IA1',
                    'JM4-IgG3',
                    'VRC26.05',
                    '2E7',
                    'sCD4',
                    'iMab',
                    'VRC26.06',
                    'F6',
                    '45-46m28',
                    '10E8',
                    'PCT64-24E',
                    'QA013.4',
                    'PGT123',
                    'VRC38.11',
                    'VRC21.01',
                    '5G2',
                    'PGT125',
                    'VRC34YD.02',
                    '3019',
                    'VRC38.05',
                    'VRC26.33',
                    'N49P6',
                    'CH17',
                    'VRC34YD.03',
                    'CH18',
                    '3074',
                    'VRC-PG19',
                    'F2',
                    'PGT131',
                    '1361',
                    'PGT136',
                    '3BNC62',
                    '561_02_12',
                    'N49P7.1',
                    'PGC14',
                    'VRC34.05',
                    'DH517',
                    'CH13',
                    'BiIA-DG',
                    'F4',
                    'PGT153',
                    'VRC18',
                    'VRC26.26',
                    'DH511.2_K3',
                    '561_01_55',
                    'PCT64-24G',
                    'CH28',
                    'PCT64-18C',
                    '561_01_18',
                    'vFP1.01',
                    'CH14',
                    'PCT64-35C',
                    'VRC26.03',
                    '8ANC131',
                    'm66-S28H',
                    'VRC06b',
                    'VRC26.17',
                    'VRC29.02',
                    '15e',
                    'VRC26.18',
                    'PGT137',
                    'QA013.19',
                    'Z13e1',
                    '10E8v4-5R-100cF',
                    'DH511.2',
                    '13I10',
                    'PGZL1.H4K3',
                    'CH03',
                    'PGDM1403',
                    'N49P19',
                    'PCT64-24A',
                    'N49P9',
                    'DRVIA6b',
                    'CH103',
                    'SF3',
                    'VRC26.02',
                    'PGDM1404',
                    'PGT124',
                    'm66-G30K',
                    'CH12',
                    'vFP20.01',
                    'HK20',
                    '1E2',
                    'DH511.3',
                    'VRC26.32',
                    '25C4b',
                    '2D4',
                    '12A21',
                    'BG8',
                    '4025',
                    'DH270.IA2',
                    'm66',
                    '4O20',
                    'PGT127',
                    'N60P22',
                    'VRC21.02',
                    'PGT143',
                    'VRC26.27',
                    'QA013.32',
                    'DRVIA1',
                    'PGDM12',
                    'N49P22',
                    '3694',
                    'VRC26.31',
                    '10-996',
                    'VRC26.12',
                    'QA013.53',
                    'DH511.4',
                    '49G2',
                    'VRC-CH34',
                    '1C2',
                    'DH511.11P',
                    'D7',
                    '2191',
                    '7K3',
                    'VRC34YD.06',
                    '7H6',
                    'QA013.2',
                    'B9',
                    'VRC38.13',
                    'PGT142',
                    '3906',
                    'VRC26.04',
                    '1D9',
                    'VRC26.07',
                    'N60P2.1',
                    'A16',
                    'C8',
                    'A14',
                    'BG1',
                    'BE10',
                    'F8',
                    'DH270.2',
                    'PCT64-35H',
                    '447-52D',
                    'VRC26.23',
                    'VRC23',
                    'VRC38.10',
                    '17b',
                    '1E1',
                    'VRC38.02',
                    'PGT141',
                    'PCT64-35D',
                    'VRC26.15',
                    'JM2',
                    'PGT158',
                    'VRC26.13',
                    'VRC26.20',
                    'VRC34YD.04',
                    'PCT64-35O',
                    'VRC26.14',
                    'PCT64-35E',
                    '3BC176',
                    '4487',
                    '3869',
                    'PCT64-18B',
                    'N60P1.1',
                    '10E8 IgG mutant 2',
                    'PGT154',
                    '4139',
                    '3904',
                    'VRC34YD.07',
                    'CH235.9',
                    'VRC-PG05',
                    'SF2',
                    '10E8 IgG mutant 5',
                    'BF8',
                    'PCDN-33A',
                    'F425',
                    '2N5',
                    'VRC06',
                    'PCT64-18D',
                    'm66.6',
                    'CD4-Ig',
                    'PGT122',
                    '1F7',
                    'CH48',
                    '10E8 IgG mutant 3',
                    'BiIA-SG',
                    'SF8',
                    'J3',
                    '12A12',
                    '1F10',
                    'KD-247',
                    'CH235',
                    'VRC23b',
                    '4508',
                    'N49P23',
                    'PCT64-35B',
                    'DRVIA7',
                    'VRC34.02',
                    'DH270.IA4',
                    'VRC16',
                    'PGT126',
                    'VRC34.07',
                    'PCT64-24H',
                    'Bi-NAb',
                    'VRC34.03',
                    '10-847',
                    'N6',
                    'BF520.1',
                    'PCT64-24F',
                    'PCT64-35N',
                    'b13',
                    '1-79',
                    '2219',
                    'VRC41.02',
                    '10-1146',
                    'PCT64-35F',
                    '4490',
                    'CH27',
                    'VRC38.04',
                    'SF12',
                    '10J4',
                    'CH104',
                    'VRC26.24',
                    'm66-S28H-G30K-S31K',
                    '4022',
                    '2557',
                    'DH270.IA3',
                    'PGDM1401',
                    '0.5γ',
                    'ACS202',
                    '10-410',
                    'PGZL1',
                    'VRC26.29',
                    '916B2',
                    'CH16',
                    '2558',
                    'QA013.3',
                    'DH270.3',
                    'H6',
                    '10-1341',
                    '48d',
                    'Y498',
                    'VRC41.01',
                    'PCT64-24B',
                    'VRC-PG20',
                    'BE7',
                    'VRC34YD.05',
                    'Tri-NAb',
                    'CH04',
                    'VRC-CH32',
                    '4121',
                    'SF10',
                    'VRC26.30',
                    '10A37',
                    'JM3',
                    'CH235.12',
                    'N60P31.1',
                    'DH563',
                    '1B2530',
                    'AIIMS-P01',
                    'DH511.12P',
                    'PGT144',
                    'NC37',
                    'CH02',
                    'VRC22.01',
                    'VRC29.01',
                    'vFP5.01',
                    'VRC26.19',
                    'PGT156',
                    '45-46m7',
                    'PGDM1406',
                    'B21',
                    'N49P7.2',
                    '10-259',
                    'N49P18.1',
                    'X5',
                    '4210',
                    'VRC26.11',
                    'VRC26.16',
                    'VRC38.14',
                    '16G6',
                    'PGT155',
                    '179NC75',
                    '10E8v4',
                    'VRC27',
                    '830A',
                    'PCT64-18F',
                    '3E3',
                    'CH44',
                    '2909',
                    'VRC26.09',
                    'PCT64-35I',
                    '4E9C',
                    'VRC28.01',
                    '2158',
                    'PCT64-35K',
                    'PGT157',
                    'vFP7.05',
                    '917B11',
                    'N49P9.1',
                    'CH106',
                    'VRC38.06',
                    '10E8 IgG mutant 1',
                    '10-1369',
                    'CH98',
                    '3881',
                    '1B5',
                    'VRC02',
                    '717G2',
                    'PCT64-35M',
                    'Bi-ScFv',
                    'HGN194',
                    'PCT64-35S',
                    'LN01',
                    'PGDM1405',
                    '1NC9',
                    'm66-S31K',
                    '3791',
                    '1H9',
                    'VRC26.10',
                    'JM4',
                    '3BNC60',
                    'DH511.1',
                    'VRC26.01',
                    'DH511.6',
                    'VRC29.04',
                    '10-1121',
                    'SF5',
                    'VRC38.12',
                    'DRVIA4',
                    'CAP257-RH1',
                    'VRC26.21',
                    'N49P11',
                    'CH105',
                    'PCDN-38B',
                    'vFP7.04',
                    '1393A',
                    'JM5',
                    'DH270.4',
                    'N49P7',
                    'vFP16.02',
                    '10-303',
                    'VRC26.28',
                    '45-46m2',
                    'm66-Y100eW',
                    'PGDM1402',
                    'VRC38.03',
                    'DRVIA3',
                    'VRC-CH30',
                    'CH45',
                    'F105',
                    'PGZL1_gVmDmJ',
                    '10-1130',
                    '3BC315',
                    'PGDM11',
                    'SF7',
                    'DH511.5',
                    'IOMA',
                    '10M6',
                    'VRC-CH33',
                    'N60P25.1',
                    'VRC34YD.01',
                    'm66-S28H-G30K',
                    'VRC38.07',
                    'PGT152',
                    '45-46m25',
                    'VRC38.09',
                    '3BNC55',
                    '2F5',
                    '2G12',
                    '3BNC117',
                    '4E10',
                    '8ANC195',
                    '10-1074',
                    '35O22',
                    'b12',
                    'CH01',
                    'DH270.1',
                    'DH270.5',
                    'DH270.6',
                    'HJ16',
                    'NIH45-46',
                    'PG9',
                    'PG16',
                    'PGDM1400',
                    'PGT121',
                    'PGT128',
                    'PGT135',
                    'PGT145',
                    'PGT151',
                    'VRC-CH31',
                    'VRC-PG04',
                    'VRC01',
                    'VRC03',
                    'VRC07',
                    'VRC13',
                    'VRC26.08',
                    'VRC26.25',
                    'VRC29.03',
                    'VRC34.01',
                    'VRC38.01']

def get_today_date_dir(output_dir):
    return_path = os.path.join(output_dir, time.strftime('%Y_%m_%d'))
    if not os.path.isdir(return_path):
        os.mkdir(return_path)
    return return_path

def write_to_csv(rows, fieldnames, filename):
    with open(filename, 'w', newline='') as f:
        print(f'writing to {filename}')
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def parse_assay_data(assay_txt):
    assay_data = {}
    with open(assay_txt, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            antibody = row['Antibody']
            antibody = antibody.replace('/', '-')
            if antibody not in assay_data:
                assay_data[antibody] = []
            virus_id = 'IAVI_C22' if row['Virus'] in ['IAVI_C22', 'MGRM_C_026'] else row['Virus']
            assay_data[antibody].append({
                        'virus_id' : virus_id,
                        'reference' : row['Reference'],
                        'pubmed_id' : row['Pubmed ID'],
                        'ic_50' : row['IC50'],
                        'ic_80' : row['IC80'],
                        'id_50' : row['ID50'],
                    })
    return assay_data

def parse_viruses_fasta(viruses_fasta):
    all_env_seqs = {}
    with open(viruses_fasta) as fasta:
        for seq in SeqIO.parse(fasta, 'fasta'):
            if seq.id in all_env_seqs:
                raise Exception(f'Duplication for seq {seq.id} in {viruses_fasta}!!!')
            all_env_seqs[seq.id] = str(seq.seq)
    return all_env_seqs

def parse_viruses_txt(viruses_txt):
    viruses_info = {}
    with open(viruses_txt, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            virus_id = row['Virus name']
            if virus_id in viruses_info:
                raise Exception(f'Duplication for virus {virus_id} in {viruses_txt}')
            if virus_id in ['IAVI_C22', 'MGRM_C_026']:#these two viruses are the same virus
                virus_id = 'IAVI_C22'
            viruses_info[virus_id] = {
                'subtype': row['Subtype'],
                'country': row['Country'],
                'year': row['Year'],
                'patient_health': row['Patient health'],
                'risk_factor': row['Risk factor'],
                'accession_number': row['Accession'],
                'tier': row['Tier'],
                'infection_stage': row['Infection stage']
            }
    return viruses_info

def remove_non_aa_characters(fasta_file, include_glycans=False, min_len=800, max_len=900):
    print(f'removing non AA characters from sequences in {fasta_file}.')
    print(f'only sequences of length in the [{min_len}-{max_len}] range will be considered.')
    seqs_dict = {}
    with open(fasta_file) as fasta:
        for seq in SeqIO.parse(fasta, 'fasta'):
            sequence_id = seq.id
            sequence_id = sequence_id.upper()#eliminate weird differences
            preprocessed_seq = ''.join([aa for aa in str(seq.seq) if aa.upper() in the_20_aa or (aa.upper() == 'O' and include_glycans)])#O means N-glycan
            if len(preprocessed_seq) < min_len or len(preprocessed_seq) > max_len:
                continue
            seqs_dict[sequence_id] = preprocessed_seq
    print(f'non AA characters were removed from {len(seqs_dict)} sequences')
    seqs_dict_df = pd.DataFrame.from_dict(seqs_dict, orient='index', columns=['sequence'])
    return seqs_dict_df

def get_sequence_alignment(alignment_filepath):
    alignment = {}
    with open(alignment_filepath) as f:
        for record in SeqIO.parse(f, "fasta"):
            # virus_id = record.id.split('.')
            # if len(virus_id) > 2:
            #     virus_id = virus_id[3]
            # else:
            #     virus_id = virus_id[0]
            # alignment[virus_id] = str(record.seq)
            alignment[record.id] = str(record.seq)
    return alignment

def generate_stratified_folds(data, n_folds=5, random_state=42):
    #we stratify by phenotype
    original_data = pd.DataFrame.copy(data)
    folds = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)

    for training_i, test_i in folds.split(data, data['subtype_phenotype']):
        test_data = original_data.iloc[test_i]
        training_data = original_data.iloc[training_i]
        yield training_data, test_data

def save_predictions(test_data, predictions, output_dir, file_prefix, std=None):
    file_path = os.path.join(output_dir, f'{file_prefix}_predictions.csv')
    if len(test_data) != len(predictions):
        raise Exception(f'predictions and true label lists must have the same length: {len(test_data)} vs {len(predictions)}')
    test_data['predictions'] = predictions
    if std is not None:
        test_data['predictions_std'] = std
    test_data.to_csv(file_path)
    
def save_best_parameters(best_params, output_dir, file_prefix):
    file_path = os.path.join(output_dir, f'{file_prefix}_best_hyperparameters.csv')
    rows = []
    column_names = ['fold'] + list(best_params[list(best_params.keys())[0]].keys())
    for fold in best_params:
        row = best_params[fold]
        row['fold'] = fold
        rows.append(row)
    write_to_csv(rows, column_names, file_path)
    
def save_best_model(model, output_dir, file_prefix):
    file_path = os.path.join(output_dir, f'{file_prefix}_best_model.pkl')
    joblib.dump(model, file_path)

def calculate_performance(testing_data, predictions):
    testing_data_copy = pd.DataFrame.copy(testing_data)
    testing_Y = testing_data_copy['phenotype']
    overall_auc = None
    overall_FPR, overall_TPR, overall_thresholds = [], [], []
    if len(testing_Y) != sum(testing_Y) and sum(testing_Y) != 0:
        overall_auc = roc_auc_score(testing_Y, predictions)
    overall_FPR, overall_TPR, overall_thresholds = roc_curve(testing_Y, predictions)
    overall_log_loss = log_loss(testing_Y, predictions, labels=[0, 1])
    overall_precs, overall_recs, overall_pr_thresholds = precision_recall_curve(testing_Y, predictions)
    overall_pr_auc = auc(overall_recs, overall_precs)
    mccs = []
    accs = []
    specs = []
    sens = []
    precs = []
    npvs = []
    for t in range(1,10,1):
        t = t/10
        mccs.append((t, matthews_corrcoef(testing_Y, predictions>=t)))
        accs.append((t, accuracy_score(testing_Y, predictions>=t)))
        sens.append((t, sensitivity_score(testing_Y, predictions>=t, average='binary')))
        specs.append((t, specificity_score(testing_Y, predictions>=t, average='binary')))
        precs.append((t, precision_score(testing_Y, predictions>=t, average='binary')))
        total_negatives = 0
        true_negatives = 0
        for i,j in zip(testing_Y, predictions):
            if j < t: 
                total_negatives += 1
                if int(i) == 0:
                    true_negatives += 1
        npvs.append((t, None if total_negatives == 0 else true_negatives/total_negatives))

    return {
        'threshold_accuracy': [[float(x), float(y)] for x,y in accs],
        'threshold_sensitivity': [[float(x), float(y)] for x,y in sens],
        'threshold_specificity': [[float(x), float(y)] for x,y in specs],
        'threshold_precision': [[float(x), float(y)] for x,y in precs],
        'threshold_negative_predictive_value': [[float(x), float(y)] if y else [float(x), None] for x,y in npvs],
        'threshold_mcc': [[float(x), float(y)] for x,y in mccs],
        'auc': float(overall_auc) if overall_auc else None,
        'pr_auc': float(overall_pr_auc),
        'log_loss': float(overall_log_loss),
        'threshold_tpr_fpr': [[float(x),float(y),float(z)] for x,y,z in zip(overall_thresholds, overall_TPR, overall_FPR)],
        'threshold_tpr_precision': [[float(x),float(y),float(z)] for x,y,z in zip(overall_pr_thresholds, overall_recs, overall_precs)],
    }

def form_language_model_input(sequences_df, fine_tuning=True, inference=False):
    model_input = []
    for _,row in sequences_df.iterrows():
        sequence = row['sequence']
        X_left = 'B' + sequence[:-1]
        X_right = sequence[1:] + 'Z'
        if fine_tuning:
            if inference:
                model_input.append((X_left, X_right, row['antibody_index']))
            else:
                model_input.append((X_left, X_right, row['antibody_index'], row['regression_weight'], row['classification_weight']))
        else: 
            model_input.append((X_left, X_right))
    if fine_tuning:
        if inference:
            return pd.DataFrame(model_input, columns=['left_input', 'right_input', 'antibody_index'])
        return pd.DataFrame(model_input, columns=['left_input', 'right_input', 'antibody_index', 'regression_weight', 'classification_weight'])
    else:
        return pd.DataFrame(model_input, columns=['left_input', 'right_input'])

def get_regions_of_interest(sequence, sites, remove_non_aa_chars):
    #replace every aa in non interesting regions with '-' character
    #the given sites are 1-indexed
    mod_sequence = []
    for i,aa in enumerate(sequence):
        if i+1 in sites:#+1 because the given sites are 1-indexed
            mod_sequence.append(aa)
        else:
            mod_sequence.append('-')
    return ''.join(mod_sequence).replace('-', '') if remove_non_aa_chars else ''.join(mod_sequence)

def map_to_hxb_cord(aligned_hxb2):
    #returns HXB2 coordinate (indexed from 1) and the HXB2 amino acid or gap
    gap_index = 0
    hxb2_index = 0
    ret_values = {}
    for i,x in enumerate(list(aligned_hxb2)):
        if x in the_20_aa:
            hxb2_index += 1
            gap_index = 0
        else:
            gap_index += 1
        # if i == ind:
        #     return hxb2_index if gap_index == 0 else f'{hxb2_index}+{gap_index}'
        if gap_index == 0:
            ret_values[i] = hxb2_index  
        else: 
            ret_values[i] = f'{hxb2_index}+{gap_index}'
    return ret_values

def get_important_sites(model_name, ic_type, cutoff, alignmet_to_hxb_cord_map, only_sites=False, model_dir='../../models'):
    hxb_idx_to_importance = {}
    for epitope in grouped_bnAbs_of_interest:
        for bnAb in grouped_bnAbs_of_interest[epitope]: 
            hxb_idx_to_importance[bnAb] = {}
            for fold in range(1,6,1):
                file_name = f"{model_dir}/{model_name}_{bnAb}_fold{fold}_ic{ic_type}_{cutoff}cutoff_best_model.pkl"
                if not os.path.isfile(file_name): continue
                model = joblib.load(file_name)
                importances = model.named_steps[model_name].feature_importances_
                important_sites = [(np.floor(x/20),importances[x]) for x in range(len(importances))]

                #accumulate all importances for each feature (i.e., each AA or gap)
                site_to_importance = {}
                for x,y in important_sites:
                    if x not in site_to_importance:
                        site_to_importance[x] = 0
                    site_to_importance[x] += y

                site_to_importance = [(k,v) for k,v in site_to_importance.items()]
                site_to_importance = sorted(site_to_importance, key=lambda x: x[1], reverse=True)
                total_importance = 0
                hxb_idx_to_importance[bnAb][fold] = []
                for site, importance in site_to_importance:
                    if importance != 0: 
                        site = int(site)
                        if only_sites:
                            hxb_idx_to_importance[bnAb][fold].append(site+1)
                        else:
                            hxb_idx_to_importance[bnAb][fold].append((alignmet_to_hxb_cord_map[site], site+1, importance))
                        total_importance += importance
    return hxb_idx_to_importance              