#!/bin/bash

python Training.py -f MNIST_VAE_AD_v1.json -c '{"RunName":"MNIST_VAE_AD_holdout0","HyperParams":{},"Dataset":{"Arguments":{"holdout":0}}}'
python Training.py -f MNIST_VAE_AD_v1.json -c '{"RunName":"MNIST_VAE_AD_holdout1","HyperParams":{},"Dataset":{"Arguments":{"holdout":1}}}'
python Training.py -f MNIST_VAE_AD_v1.json -c '{"RunName":"MNIST_VAE_AD_holdout2","HyperParams":{},"Dataset":{"Arguments":{"holdout":2}}}'
python Training.py -f MNIST_VAE_AD_v1.json -c '{"RunName":"MNIST_VAE_AD_holdout3","HyperParams":{},"Dataset":{"Arguments":{"holdout":3}}}'
python Training.py -f MNIST_VAE_AD_v1.json -c '{"RunName":"MNIST_VAE_AD_holdout4","HyperParams":{},"Dataset":{"Arguments":{"holdout":4}}}'
python Training.py -f MNIST_VAE_AD_v1.json -c '{"RunName":"MNIST_VAE_AD_holdout5","HyperParams":{},"Dataset":{"Arguments":{"holdout":5}}}'
python Training.py -f MNIST_VAE_AD_v1.json -c '{"RunName":"MNIST_VAE_AD_holdout6","HyperParams":{},"Dataset":{"Arguments":{"holdout":6}}}'
python Training.py -f MNIST_VAE_AD_v1.json -c '{"RunName":"MNIST_VAE_AD_holdout7","HyperParams":{},"Dataset":{"Arguments":{"holdout":7}}}'
python Training.py -f MNIST_VAE_AD_v1.json -c '{"RunName":"MNIST_VAE_AD_holdout8","HyperParams":{},"Dataset":{"Arguments":{"holdout":8}}}'
python Training.py -f MNIST_VAE_AD_v1.json -c '{"RunName":"MNIST_VAE_AD_holdout9","HyperParams":{},"Dataset":{"Arguments":{"holdout":9}}}'
