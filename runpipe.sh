#!/bin/bash
dvc run --force -n data_load \
        -d data/imoveis_pocos.csv \
        -d src/data_load.py \
        -o pkl/raw_data.pkl \
        -p data_load\
        python src/data_load.py \
        --config=params.yaml 

dvc run --force -n feature_transform \
        -d pkl/raw_data.pkl \
        -d src/feature_transform.py \
        -o pkl/features.pkl \
        -o pkl/targets.pkl \
        -p feature_transform \
        python src/feature_transform.py \
        --config=params.yaml 

dvc run --force -n splitting \
        -d pkl/features.pkl \
        -d pkl/targets.pkl \
        -d src/splitting.py \
        -o pkl/features_train.pkl \
        -o pkl/features_test.pkl \
        -o pkl/targets_train.pkl \
        -o pkl/targets_test.pkl \
        -p splitting \
        python src/splitting.py \
        --config=params.yaml


dvc run --force -n model_training \
        -d pkl/features_train.pkl \
        -d pkl/targets_train.pkl \
        -d src/model_training.py \
        -o pkl/model.pkl \
        -p model \
        python src/model_training.py \
        --config=params.yaml
