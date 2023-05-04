#!/bin/bash

python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout0","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":0}}}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout1","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":1}}}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout2","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":2}}}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout3","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":3}}}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout4","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":4}}}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout5","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":5}}}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout6","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":6}}}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout7","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":7}}}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout8","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":8}}}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout9","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":9}}}'
