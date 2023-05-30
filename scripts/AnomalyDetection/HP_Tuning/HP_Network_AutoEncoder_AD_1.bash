#!/bin/bash


python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout0_Netv2_FC32_FT8","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":0}}}' -n '{"FCUnits":32,"Filters":8}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout1_Netv2_FC32_FT8","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":1}}}' -n '{"FCUnits":32,"Filters":8}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout5_Netv2_FC32_FT8","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":5}}}' -n '{"FCUnits":32,"Filters":8}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout7_Netv2_FC32_FT8","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":7}}}' -n '{"FCUnits":32,"Filters":8}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout9_Netv2_FC32_FT8","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":9}}}' -n '{"FCUnits":32,"Filters":8}'

python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout0_Netv2_FC64_FT8","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":0}}}' -n '{"FCUnits":64,"Filters":8}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout1_Netv2_FC64_FT8","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":1}}}' -n '{"FCUnits":64,"Filters":8}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout5_Netv2_FC64_FT8","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":5}}}' -n '{"FCUnits":64,"Filters":8}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout7_Netv2_FC64_FT8","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":7}}}' -n '{"FCUnits":64,"Filters":8}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout9_Netv2_FC64_FT8","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":9}}}' -n '{"FCUnits":64,"Filters":8}'

python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout0_Netv2_FC64_FT16","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":0}}}' -n '{"FCUnits":64,"Filters":16}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout1_Netv2_FC64_FT16","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":1}}}' -n '{"FCUnits":64,"Filters":16}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout5_Netv2_FC64_FT16","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":5}}}' -n '{"FCUnits":64,"Filters":16}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout7_Netv2_FC64_FT16","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":7}}}' -n '{"FCUnits":64,"Filters":16}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout9_Netv2_FC64_FT16","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":9}}}' -n '{"FCUnits":64,"Filters":16}'

python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout0_Netv2_FC128_FT16","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":0}}}' -n '{"FCUnits":128,"Filters":16}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout1_Netv2_FC128_FT16","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":1}}}' -n '{"FCUnits":128,"Filters":16}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout5_Netv2_FC128_FT16","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":5}}}' -n '{"FCUnits":128,"Filters":16}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout7_Netv2_FC128_FT16","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":7}}}' -n '{"FCUnits":128,"Filters":16}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout9_Netv2_FC128_FT16","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":9}}}' -n '{"FCUnits":128,"Filters":16}'

python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout0_Netv2_FC128_FT32","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":0}}}' -n '{"FCUnits":128,"Filters":32}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout1_Netv2_FC128_FT32","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":1}}}' -n '{"FCUnits":128,"Filters":32}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout5_Netv2_FC128_FT32","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":5}}}' -n '{"FCUnits":128,"Filters":32}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout7_Netv2_FC128_FT32","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":7}}}' -n '{"FCUnits":128,"Filters":32}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout9_Netv2_FC128_FT32","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":9}}}' -n '{"FCUnits":128,"Filters":32}'

python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout0_Netv2_FC256_FT32","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":0}}}' -n '{"FCUnits":256,"Filters":32}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout1_Netv2_FC256_FT32","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":1}}}' -n '{"FCUnits":256,"Filters":32}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout5_Netv2_FC256_FT32","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":5}}}' -n '{"FCUnits":256,"Filters":32}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout7_Netv2_FC256_FT32","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":7}}}' -n '{"FCUnits":256,"Filters":32}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout9_Netv2_FC256_FT32","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":9}}}' -n '{"FCUnits":256,"Filters":32}'

python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout0_Netv2_FC256_FT64","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":0}}}' -n '{"FCUnits":256,"Filters":64}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout1_Netv2_FC256_FT64","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":1}}}' -n '{"FCUnits":256,"Filters":64}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout5_Netv2_FC256_FT64","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":5}}}' -n '{"FCUnits":256,"Filters":64}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout7_Netv2_FC256_FT64","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":7}}}' -n '{"FCUnits":256,"Filters":64}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout9_Netv2_FC256_FT64","HyperParams":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":9}}}' -n '{"FCUnits":256,"Filters":64}'
