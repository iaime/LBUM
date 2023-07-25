FROM python:3.9.17-slim-bookworm 
RUN mkdir /home/inputs
RUN mkdir /home/outputs
COPY ./scripts /home/scripts
WORKDIR /home
RUN pip3 install biopython==1.79
RUN pip3 install pandas==1.4.2
CMD python ./scripts/preprocess_input_fasta.py -a ./inputs/$NON_ALIGNED_ENV_FASTA

FROM ddiez/mafft
USER 0
RUN mkdir /home/outputs
RUN mkdir /home/inputs
RUN mkdir /home/datasets
WORKDIR /home
CMD mafft --add ./outputs/preprocessed_$NON_ALIGNED_ENV_FASTA --keeplength ./datasets/$CATNAP_ALIGNMENT > ./outputs/$ALIGNED_ENV_FASTA

FROM python:3.9.17-slim-bookworm
COPY ./scripts /home/scripts
COPY ./final_trained_models /home/final_trained_models
COPY ./pretrained_models /home/pretrained_models
COPY requirements.txt /home
RUN mkdir /home/outputs
RUN mkdir /home/inputs
WORKDIR /home
RUN pip3 install -r requirements.txt
CMD python ./scripts/predict.py -o ./outputs -d ./outputs/preprocessed_$NON_ALIGNED_ENV_CSV -a ./outputs/$ALIGNED_ENV_FASTA -p $PREFIX -b $BNABS -m $MODELS