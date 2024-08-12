# Predicting HIV-1 resistance to bnAbs

This repository accompanies the following paper: "Learning patterns of HIV-1 co-resistance to broadly neutralizing antibodies with reduced subtype bias using multi-task learning" 
https://doi.org/10.1101/2023.09.28.559724

Three types of models were developed in this study: RF, GBM and LBUM. They can predict HIV-1 resistance to the following 33 bnAbs: 2F5, 4E10, 8ANC195, 35O22, PGT151, VRC34.01, 3BNC117, b12, VRC01, VRC03, VRC07, VRC13, HJ16, NIH45-46, VRC-CH31, VRC-PG04, 2G12, PGT128, PGT121, 10-1074, PGT135, DH270.1, DH270.5, DH270.6, VRC29.03, PG9, PG16, CH01, PGT145, VRC26.25, VRC26.08, VRC38.01, and PGDM1400.

The models output the probability of resistance (i.e., a score in the \[0-1\] range, where 1 means reistance and 0 means sensitive). 

To use the models, we recommend using a conda environment with Python 3.9. Required packages are given in requirements.txt. After setting up your environment, the general workflow should be as follows:
  
1. Download/clone this GitHub repository

2. Download models from the following Zenodo at https://zenodo.org/doi/10.5281/zenodo.13286435, and put the folder in the downloaded/cloned repository.

3. Preprocess your non-aligned sequences:
     ```shellscript
     python ./scripts/preprocess_input_fasta.py -a PATH_TO_NON_ALIGNED_ENV_FASTA
     ```
     The script will generate two files: a csv file and a fasta file containing preprocessed but not aligned sequences.

4. Using tools such as MAFFT, align your preprocessed sequences to reference_catnap_alignment.fasta found in this repository, making sure that the length of the alignment is kept (i.e., for MAFFT, please specify ```--keeplength```). RF and GBM expect CATNAP alignment's length.

5. Run the models using the given predict.py script. Please specify the following options:
     ```--output_dir```: path to where you want output files to be saved. There will be one output file per run.
   
     ```--prefix```: unique prefix that will be added to final output filenames. If the prefix is not unique, files in the output folder may be overwritten.
   
     ```--bnAb```: the bnAb of interest.
   
     ```--models```: comma-separated list of models you want to use. Available options are GBM, RF, LBUM, ENS --for ensemble of the three--, and 'best' for the best models for the bnAb in question. See 'best_models.json'. Best models were determined based on AUC. Please see the manuscript for details. When ENS is not specified but all three models are specified, ENS will be included in the output anyway.
   
     ```--preprocessed_nonaligned_csv```: path to the csv file containing preprocessed but not aligned input sequences. This file was generated when you ran ./scripts/preprocess_input_fasta.py
   
     ```--preprocessed_aligned_fasta```: path to the fasta file containing preprocessed and aligned input sequences. This is the file you generated after running some alignment tool.
   
     ```--ic```: either 50 or 80 to specify whether you're interested in IC50-based models or IC80-based models. 1 ug/ml threshold was used for IC80 while 50 ug/ml threshold was used for IC50. See manuscript for more detail.
     
If you use our methods in your work, please cite the following manuscript: https://doi.org/10.1101/2023.09.28.559724

