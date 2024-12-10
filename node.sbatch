#!/bin/bash

    source $HOME/miniconda/bin/activate
    conda activate testing

    python cnn.py     --batch_size 1024     --learning_rate 0.02     --model_name ResNet18     --dataset_name CIFAR10     --optimizer_name SGD     --epochs 20
    