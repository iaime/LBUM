import argparse, os
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord 

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
             
def remove_non_aa_characters(fasta_file, min_len=800, max_len=900):
    print(f'removing non AA characters from sequences in {fasta_file}.')
    print(f'only sequences of length in the [{min_len}-{max_len}] range will be considered.')
    seqs_dict = {}
    with open(fasta_file) as fasta:
        for seq in SeqIO.parse(fasta, 'fasta'):
            sequence_id = seq.id
            sequence_id = sequence_id.upper()#eliminate weird differences
            preprocessed_seq = ''.join([aa for aa in str(seq.seq) if aa.upper() in the_20_aa])
            if len(preprocessed_seq) < min_len or len(preprocessed_seq) > max_len:
                continue
            seqs_dict[sequence_id] = preprocessed_seq
    print(f'non AA characters were removed from {len(seqs_dict)} sequences')
    seqs_dict_df = pd.DataFrame([(k,v) for k,v in seqs_dict.items()], columns=['virus_id', 'sequence'])
    return seqs_dict_df

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--non_aligned_fasta', action='store', type=str, required=True)
    parser.add_argument('-o', '--output_dir', action='store', type=str, required=False, default='./outputs')

    args = parser.parse_args()

    _, file_name = os.path.split(os.path.splitext(args.non_aligned_fasta)[0])
    data = remove_non_aa_characters(args.non_aligned_fasta)
    data.to_csv(os.path.join(args.output_dir, f'preprocessed_{file_name}.csv'))
    output_fasta = os.path.join(args.output_dir, f'preprocessed_{file_name}.fasta')
    SeqIO.write([SeqRecord(Seq(row['sequence']), id=row['virus_id'], description='') for _,row in data.iterrows()], output_fasta, 'fasta')