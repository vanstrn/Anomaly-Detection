#!/bin/bash


python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout0_Netv4_FC32","HyperParams":{"NetworkConfig":"AE_MNIST_v3.json"},"Dataset":{"Arguments":{"holdout":0}}}' -n '{"FCUnits":32}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout1_Netv4_FC32","HyperParams":{"NetworkConfig":"AE_MNIST_v3.json"},"Dataset":{"Arguments":{"holdout":1}}}' -n '{"FCUnits":32}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout5_Netv4_FC32","HyperParams":{"NetworkConfig":"AE_MNIST_v3.json"},"Dataset":{"Arguments":{"holdout":5}}}' -n '{"FCUnits":32}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout7_Netv4_FC32","HyperParams":{"NetworkConfig":"AE_MNIST_v3.json"},"Dataset":{"Arguments":{"holdout":7}}}' -n '{"FCUnits":32}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout9_Netv4_FC32","HyperParams":{"NetworkConfig":"AE_MNIST_v3.json"},"Dataset":{"Arguments":{"holdout":9}}}' -n '{"FCUnits":32}'

python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout0_Netv4_FC64","HyperParams":{"NetworkConfig":"AE_MNIST_v3.json"},"Dataset":{"Arguments":{"holdout":0}}}' -n '{"FCUnits":64}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout1_Netv4_FC64","HyperParams":{"NetworkConfig":"AE_MNIST_v3.json"},"Dataset":{"Arguments":{"holdout":1}}}' -n '{"FCUnits":64}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout5_Netv4_FC64","HyperParams":{"NetworkConfig":"AE_MNIST_v3.json"},"Dataset":{"Arguments":{"holdout":5}}}' -n '{"FCUnits":64}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout7_Netv4_FC64","HyperParams":{"NetworkConfig":"AE_MNIST_v3.json"},"Dataset":{"Arguments":{"holdout":7}}}' -n '{"FCUnits":64}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout9_Netv4_FC64","HyperParams":{"NetworkConfig":"AE_MNIST_v3.json"},"Dataset":{"Arguments":{"holdout":9}}}' -n '{"FCUnits":64}'

python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout0_Netv4_FC128","HyperParams":{"NetworkConfig":"AE_MNIST_v3.json"},"Dataset":{"Arguments":{"holdout":0}}}' -n '{"FCUnits":128}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout1_Netv4_FC128","HyperParams":{"NetworkConfig":"AE_MNIST_v3.json"},"Dataset":{"Arguments":{"holdout":1}}}' -n '{"FCUnits":128}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout5_Netv4_FC128","HyperParams":{"NetworkConfig":"AE_MNIST_v3.json"},"Dataset":{"Arguments":{"holdout":5}}}' -n '{"FCUnits":128}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout7_Netv4_FC128","HyperParams":{"NetworkConfig":"AE_MNIST_v3.json"},"Dataset":{"Arguments":{"holdout":7}}}' -n '{"FCUnits":128}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout9_Netv4_FC128","HyperParams":{"NetworkConfig":"AE_MNIST_v3.json"},"Dataset":{"Arguments":{"holdout":9}}}' -n '{"FCUnits":128}'

python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout0_Netv4_FC256","HyperParams":{"NetworkConfig":"AE_MNIST_v3.json"},"Dataset":{"Arguments":{"holdout":0}}}' -n '{"FCUnits":256}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout1_Netv4_FC256","HyperParams":{"NetworkConfig":"AE_MNIST_v3.json"},"Dataset":{"Arguments":{"holdout":1}}}' -n '{"FCUnits":256}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout5_Netv4_FC256","HyperParams":{"NetworkConfig":"AE_MNIST_v3.json"},"Dataset":{"Arguments":{"holdout":5}}}' -n '{"FCUnits":256}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout7_Netv4_FC256","HyperParams":{"NetworkConfig":"AE_MNIST_v3.json"},"Dataset":{"Arguments":{"holdout":7}}}' -n '{"FCUnits":256}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout9_Netv4_FC256","HyperParams":{"NetworkConfig":"AE_MNIST_v3.json"},"Dataset":{"Arguments":{"holdout":9}}}' -n '{"FCUnits":256}'
