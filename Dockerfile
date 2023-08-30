# FROM ubuntu:latest
# RUN apt-get update && apt-get install -y python3 python3-pip
# RUN mkdir /home/inputs
# RUN mkdir /home/outputs
# COPY ./scripts /home/scripts
# WORKDIR /home
# RUN pip3 install biopython==1.79
# RUN pip3 install pandas==1.4.2
# CMD python3 ./scripts/preprocess_input_fasta.py -a ./inputs/$NON_ALIGNED_ENV_FASTA

# FROM ddiez/mafft
# USER 0
# RUN mkdir /home/outputs
# RUN mkdir /home/inputs
# COPY reference_catnap_alignment.fasta /home
# WORKDIR /home
# CMD mafft --add ./outputs/preprocessed_$NON_ALIGNED_ENV_FASTA --keeplength ./reference_catnap_alignment.fasta > ./outputs/aligned_preprocessed_$NON_ALIGNED_ENV_FASTA

FROM ubuntu:latest
COPY ./scripts /home/scripts
COPY ./final_trained_models /home/final_trained_models
COPY ./pretrained_models /home/pretrained_models
COPY requirements.txt /home
RUN mkdir /home/outputs
RUN mkdir /home/inputs
WORKDIR /home
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install -r requirements.txt
CMD python3 ./scripts/predict.py -o ./outputs -a ./outputs/preprocessed_$NON_ALIGNED_ENV_FASTA -p $PREFIX -b $BNABS -m $MODELS