# Predicting HIV-1 resistance to bnAbs

Three types of models were developed in this study: RF, GBM and LBUM. They can predict HIV-1 resistance to the following 33 bnAbs: 2F5, 4E10, 8ANC195, 35O22, PGT151, VRC34.01, 3BNC117, b12, VRC01, VRC03, VRC07, VRC13, HJ16, NIH45-46, VRC-CH31, VRC-PG04, 2G12, PGT128, PGT121, 10-1074, PGT135, DH270.1, DH270.5, DH270.6, VRC29.03, PG9, PG16, CH01, PGT145, VRC26.25, VRC26.08, VRC38.01, and PGDM1400.

The models output the probability of resistance (i.e., a score in the \[0-1\] range, where 1 means reistance and 0 means sensitive). 

To use the models, please follow the following steps:

1. If you haven't already, install Docker and start it up
  
2. Download/clone this GitHub repository

3. Under the main folder, create “inputs” and “outputs” folders

4. Add the fasta file containing your sequences to the “inputs” folder 

5. Edit the LBUM.yaml file. Specifically change the following entries:
     - PREFIX: unique prefix that will be appended to final output files. If the prefix is not unique, files in the outputs folder may be overwritten.
     - BNABS: comma-separated (no space) list of bnAbs of interest
     - MODELS: comma-separated (no space) list of models you want to use (GBM,RF,LBUM).
     - NON_ALIGNED_ENV_FASTA: the name of the fasta file containing the input sequences (just the name and not the file path).
     - All the way down under services/MODELS, set "image" to either iaime/lbum_amd64 or iaime/lbum_arm64 depending on your platform.
  
6. Below are the three steps involved in running the models, along with corresponding docker commands. Please make sure you're under the main folder of the downloaded/cloned repository.
    - Preprocessing
     ```shellscript
     docker-compose -f LBUM.yaml up PREPROCESS
     ```
    - Aligning sequences using MAFFT
     ```shellscript
     docker-compose -f LBUM.yaml up MAFFT
     ```
    - Running the models
     ```shellscript
     docker-compose -f LBUM.yaml up MODELS
     ```

7. Analyze the outputs in the outputs folder

8. Cite our methods :)

Potential errors:

  **Error**: ! MODELS The requested image's platform (linux/arm64/v8) does not match the detected host platform (linux/amd64/v3) and no specific platform was requested. **Solution**: Please make sure you are using the right iaime/lbum image in the MODELS service in LBUM.yaml.

  **Error**: failed to register layer: sync /var/lib/docker/image/overlay2/layerdb/tmp/write-set-2840780088/diff: input/output error. **Solution**: Please make sure you have enough space on your machine.
