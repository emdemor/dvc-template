#!/bin/bash
dvc run --force -n feature_transform \
        -d data/imoveis_pocos.csv \
        -d source/stg01_feature_transform.py \
        -o stages/X.csv \
        -o stages/y.csv \
        -o stages/feature_transform_pipeline.pkl \
        -p feature_transform\
        python source/stg01_feature_transform.py \
        --config=params.yaml 


dvc run --force -n splitting \
        -d stages/X.csv \
        -d stages/y.csv \
        -d source/stg02_splitting.py \
        -o stages/X_train.csv \
        -o stages/y_train.csv \
        -o stages/X_test.csv \
        -o stages/y_test.csv \
        -p splitting \
        python source/stg02_splitting.py \
        --config=params.yaml


dvc run --force -n model_training \
        -d stages/X_train.csv \
        -d stages/y_train.csv \
        -d source/stg03_model_training.py \
        -o stages/model.pkl \
        -p model \
        python source/stg03_model_training.py \
        --config=params.yaml


dvc run --force -n model_testing \
        -d stages/model.pkl \
        -d stages/X_test.csv \
        -d stages/y_test.csv \
        -d source/stg04_model_testing.py \
        -m stages/metrics.json \
        -p test \
        python source/stg04_model_testing.py \
        --config=params.yaml
