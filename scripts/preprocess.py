import argparse, json, os, sys
import pandas as pd
import statistics
from utils import  (    remove_non_aa_characters, 
                        parse_assay_data, 
                        parse_viruses_txt, 
                        subtract_df_from_another, 
                        write_to_csv, combos,
                        all_antibodies)

def merge_sequences_and_phenotypes(sequences_df, phenotypes_df, phenotype_data_filepath):
    sequences_data = {}
    phenotypes_data = {}
    for virus_id, seq in sequences_df.iterrows():
        sequences_data[virus_id.split('.')[3]] = seq['sequence']
    for virus_id, row in phenotypes_df.iterrows():
        phenotypes_data[virus_id] = (row['subtype'], row['phenotype'], row['ic50'], row['right_censored'])

    intersection_viruses = list(set(sequences_data.keys()).intersection(set(phenotypes_data.keys())))
    seq_to_phen_dict = {}
    for virus_id in intersection_viruses:
        seq_to_phen_dict[virus_id] = (sequences_data[virus_id], phenotypes_data[virus_id][0], phenotypes_data[virus_id][1], phenotypes_data[virus_id][2], phenotypes_data[virus_id][3])
    
    data = pd.DataFrame.from_dict(seq_to_phen_dict, orient='index', columns=['sequence', 'subtype', 'phenotype', 'ic50', 'right_censored'])
    data.to_csv(phenotype_data_filepath, index_label='virus_id')

def compile_ic50_and_viral_info(assay_data, virus_data, bnAb_filename, bnAb):
    fieldnames = [  'virus_id', 
                    'ic50', 
                    'right_censored',
                    'pubmed_id', 
                    'reference', 
                    'subtype',
                    'country',
                    'sample_year',
                    'patient_health',
                    'risk_factor',
                    'accession_number',
                    'tier',
                    'infection_stage', 
                ]
    rows_to_save = []
    for data_point in assay_data[bnAb]:
        row = {}
        ic_value = data_point['ic_50']
        virus_id = data_point['virus_id']
        if virus_id not in virus_data:
            continue
        # We transform all <x values to x and we keep >x values if x is not too small
        if ic_value == '':
            continue
        if ic_value[0] == '>':
            if float(ic_value[1:]) < 10: #we exclude >x for small values of x because they're likely meaningless
                # print(ic_value)
                continue
            row['ic50'] = float(ic_value[1:])
            row['right_censored'] = 1
        elif ic_value[0] == '<':
            row['ic50'] = float(ic_value[1:])
            row['right_censored'] = 0
        else:
            row['ic50'] = float(ic_value)
            row['right_censored'] = 0
        row['virus_id'] = virus_id
        row['pubmed_id'] = data_point['pubmed_id']
        row['reference'] = data_point['reference']
        sample_year = virus_data[virus_id]['year']
        if sample_year == '': continue
        row['sample_year'] = int(sample_year)
        row['subtype'] = virus_data[virus_id]['subtype']
        for k in [
                    'country',
                    'patient_health',
                    'risk_factor',
                    'accession_number',
                    'tier',
                    'infection_stage'
                ]:
            row[k] = virus_data[virus_id][k]
        rows_to_save.append(row)
    write_to_csv(rows_to_save, fieldnames, bnAb_filename)

def preprocess_phenotypes(bnAb_raw_ics_filepath, sensitivity_cutoff):
    data_df = pd.read_csv(bnAb_raw_ics_filepath) 
    filtered_data_df = data_df.drop_duplicates(['pubmed_id', 'virus_id', 'ic50'])#this is to remove possible duplicated reporting
    right_censored_df = filtered_data_df[filtered_data_df['right_censored'] == 1]
    right_censored_ic50 = right_censored_df['ic50'].groupby(right_censored_df['virus_id']).max()
    right_censored_ic50 = right_censored_ic50.to_frame()
    right_censored_ic50['subtype'] = [right_censored_df[right_censored_df['virus_id'] == x]['subtype'].values[0] for x,_ in right_censored_ic50.iterrows()]
    right_censored_ic50['right_censored'] = [1 for _ in range(len(right_censored_ic50))]
    right_censored_ic50['phenotype'] = [1 for _ in range(len(right_censored_ic50))]

    data = None
    not_censored_df = filtered_data_df[filtered_data_df['right_censored'] == 0]
    if len(not_censored_df) == 0:
        data = right_censored_ic50
    else:
        data = not_censored_df['ic50'].groupby(not_censored_df['virus_id'], group_keys=False).apply(statistics.geometric_mean)
        data = data.to_frame()
        data['subtype'] = [not_censored_df[not_censored_df['virus_id'] == x]['subtype'].values[0] for x,_ in data.iterrows()]
        data['right_censored'] = [0 for _ in range(len(data))]
        data['phenotype'] = [1 if x > sensitivity_cutoff else 0 for x in data['ic50']]
        data = pd.concat([data, right_censored_ic50])
    return data
    
def save_pretraining_data(params, output_dir):
    #load CATNAP sequences (fine-tuning data) and Los Alamos sequences (pretraining data)
    #we only accept sequences whose length is within [800, 900]
    catnap_sequences = remove_non_aa_characters(params['raw_data']['catnap_fasta'])
    rio_sequences = remove_non_aa_characters(params['raw_data']['rio_fasta'])
    pangea_sequences = remove_non_aa_characters(params['raw_data']['typewriter_fasta'])
    los_alamos_sequences = remove_non_aa_characters(params['raw_data']['los_alamos_fasta'])
    catnap_sequences.drop_duplicates(subset='sequence', inplace=True)
    los_alamos_sequences.drop_duplicates(subset='sequence', inplace=True)
    rio_sequences.drop_duplicates(subset='sequence', inplace=True)
    print(f'loaded {len(catnap_sequences)} CATNAP sequences') 
    print(f'loaded {len(los_alamos_sequences)} Los Alamos sequences')
    print(f'loaded {len(rio_sequences)} RIO sequences')
    print(f'loaded {len(pangea_sequences)} PANGEA sequences')

    #make sure there is no overlap between pretraining data and catnap data
    pretraining_sequences = subtract_df_from_another(los_alamos_sequences, catnap_sequences, 'sequence')
    pretraining_sequences = pd.concat([pretraining_sequences, catnap_sequences, rio_sequences, pangea_sequences])
    print('total pretraining sequences:', len(pretraining_sequences))

    # #double check that there is no intersection between pretraining data and fine-tuning data
    # pretraining_viruses = set(pretraining_sequences.index)
    # fine_tuning_viruses = set(catnap_sequences.index)
    # pretraining_seqs = set(pretraining_sequences['sequence'])
    # fine_tuning_seqs = set(catnap_sequences['sequence'])
    # assert len(pretraining_viruses.intersection(fine_tuning_viruses)) == 0
    # assert len(pretraining_seqs.intersection(fine_tuning_seqs)) == 0

    #save data for pretraining
    pretraining_seqs_filepath = os.path.join(output_dir, 'pretraining_sequences.pkl')   
    pretraining_sequences.to_pickle(pretraining_seqs_filepath)

    #check sequences' lengths
    pretraining_lengths = [len(x) for x in pretraining_sequences['sequence']]
    fine_tuning_lengths = [len(x) for x in catnap_sequences['sequence']]
    print('pretraining sequences -', 'max sequence length:', max(pretraining_lengths), 'min sequence length:', min(pretraining_lengths))
    print('fine-tuning sequences -', 'max sequence length:', max(fine_tuning_lengths), 'min sequence length:', min(fine_tuning_lengths))
    return pretraining_sequences, catnap_sequences

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--config', action='store', type=str, required=True)
    parser.add_argument('-s', '--sensitivity_cutoff', action='store', type=int, required=False, default=50)
    args = parser.parse_args()
    with open(args.config) as params_json:
        params = json.load(params_json)

    assay_data = parse_assay_data(params['raw_data']['neutralization_assays_txt'])
    virus_data = parse_viruses_txt(params['raw_data']['viruses_info_txt'])

    proprocessed_data_dir = params['preprocessed_data_dir']
    
    bnAbs_list = all_antibodies + combos
    #compile IC50 values and viral information into one CSV file
    bnAb_to_phen_dict = {}
    print(f'compiling data on antibodies and viruses...')
    for bnAb in bnAbs_list:
        bnAb_dir = os.path.join(proprocessed_data_dir, bnAb)
        if not os.path.isdir(bnAb_dir):
            os.mkdir(bnAb_dir)
        bnAb_raw_ics_filepath = os.path.join(bnAb_dir, f'{bnAb}_ic50_and_viral_info.csv')
        compile_ic50_and_viral_info(assay_data, virus_data, bnAb_raw_ics_filepath, bnAb)
        phenotypes = preprocess_phenotypes(bnAb_raw_ics_filepath, args.sensitivity_cutoff)
        bnAb_to_phen_dict[bnAb] = phenotypes

    _, fine_tuning_sequences = save_pretraining_data(params, proprocessed_data_dir)

    print(f'saving phenotypes...')
    for bnAb in bnAbs_list:
        bnAb_dir = os.path.join(proprocessed_data_dir, bnAb)
        phenotype_data_filepath = os.path.join(bnAb_dir, f'{bnAb}_all_phenotypes.csv')
        merge_sequences_and_phenotypes(fine_tuning_sequences, bnAb_to_phen_dict[bnAb], phenotype_data_filepath)
