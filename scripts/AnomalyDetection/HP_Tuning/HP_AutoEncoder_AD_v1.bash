#!/bin/bash


python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout0_LS4","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":0}}}' -n '{"LatentSize":4}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout1_LS4","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":1}}}' -n '{"LatentSize":4}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout5_LS4","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":5}}}' -n '{"LatentSize":4}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout7_LS4","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":7}}}' -n '{"LatentSize":4}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout9_LS4","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":9}}}' -n '{"LatentSize":4}'

python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout0_LS8","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":0}}}' -n '{"LatentSize":8}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout1_LS8","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":1}}}' -n '{"LatentSize":8}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout5_LS8","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":5}}}' -n '{"LatentSize":8}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout7_LS8","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":7}}}' -n '{"LatentSize":8}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout9_LS8","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":9}}}' -n '{"LatentSize":8}'

python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout0_LS16","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":0}}}' -n '{"LatentSize":16}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout1_LS16","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":1}}}' -n '{"LatentSize":16}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout5_LS16","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":5}}}' -n '{"LatentSize":16}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout7_LS16","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":7}}}' -n '{"LatentSize":16}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout9_LS16","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":9}}}' -n '{"LatentSize":16}'

python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout0_LS32","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":0}}}' -n '{"LatentSize":32}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout1_LS32","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":1}}}' -n '{"LatentSize":32}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout5_LS32","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":5}}}' -n '{"LatentSize":32}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout7_LS32","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":7}}}' -n '{"LatentSize":32}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout9_LS32","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":9}}}' -n '{"LatentSize":32}'

python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout0_LS64","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":0}}}' -n '{"LatentSize":64}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout1_LS64","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":1}}}' -n '{"LatentSize":64}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout5_LS64","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":5}}}' -n '{"LatentSize":64}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout7_LS64","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":7}}}' -n '{"LatentSize":64}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout9_LS64","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":9}}}' -n '{"LatentSize":64}'

# python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout0_BS64","NetworkHPs":{"BatchSize":64},"Dataset":{"Arguments":{"holdout":0}}}'
# python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout1_BS64","NetworkHPs":{"BatchSize":64},"Dataset":{"Arguments":{"holdout":1}}}'
# python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout5_BS64","NetworkHPs":{"BatchSize":64},"Dataset":{"Arguments":{"holdout":5}}}'
# python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout7_BS64","NetworkHPs":{"BatchSize":64},"Dataset":{"Arguments":{"holdout":7}}}'
# python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout9_BS64","NetworkHPs":{"BatchSize":64},"Dataset":{"Arguments":{"holdout":9}}}'

python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout0_BS128","NetworkHPs":{"BatchSize":128},"Dataset":{"Arguments":{"holdout":0}}}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout1_BS128","NetworkHPs":{"BatchSize":128},"Dataset":{"Arguments":{"holdout":1}}}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout5_BS128","NetworkHPs":{"BatchSize":128},"Dataset":{"Arguments":{"holdout":5}}}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout7_BS128","NetworkHPs":{"BatchSize":128},"Dataset":{"Arguments":{"holdout":7}}}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout9_BS128","NetworkHPs":{"BatchSize":128},"Dataset":{"Arguments":{"holdout":9}}}'

python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout0_DO10","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":0}}}' -n '{"DropoutRate":0.10}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout1_DO10","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":1}}}' -n '{"DropoutRate":0.10}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout5_DO10","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":5}}}' -n '{"DropoutRate":0.10}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout7_DO10","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":7}}}' -n '{"DropoutRate":0.10}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout9_DO10","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":9}}}' -n '{"DropoutRate":0.10}'

python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout0_DO00","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":0}}}' -n '{"DropoutRate":0.00}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout1_DO00","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":1}}}' -n '{"DropoutRate":0.00}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout5_DO00","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":5}}}' -n '{"DropoutRate":0.00}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout7_DO00","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":7}}}' -n '{"DropoutRate":0.00}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout9_DO00","NetworkHPs":{},"Dataset":{"Arguments":{"holdout":9}}}' -n '{"DropoutRate":0.00}'

python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout0_Netv2","NetworkHPs":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":0}}}' -n '{}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout1_Netv2","NetworkHPs":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":1}}}' -n '{}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout5_Netv2","NetworkHPs":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":5}}}' -n '{}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout7_Netv2","NetworkHPs":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":7}}}' -n '{}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout9_Netv2","NetworkHPs":{"NetworkConfig":"AE_MNIST_v2.json"},"Dataset":{"Arguments":{"holdout":9}}}' -n '{}'

python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout0_Netv3","NetworkHPs":{"NetworkConfig":"AE_MNIST_v3.json"},"Dataset":{"Arguments":{"holdout":0}}}' -n '{}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout1_Netv3","NetworkHPs":{"NetworkConfig":"AE_MNIST_v3.json"},"Dataset":{"Arguments":{"holdout":1}}}' -n '{}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout5_Netv3","NetworkHPs":{"NetworkConfig":"AE_MNIST_v3.json"},"Dataset":{"Arguments":{"holdout":5}}}' -n '{}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout7_Netv3","NetworkHPs":{"NetworkConfig":"AE_MNIST_v3.json"},"Dataset":{"Arguments":{"holdout":7}}}' -n '{}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout9_Netv3","NetworkHPs":{"NetworkConfig":"AE_MNIST_v3.json"},"Dataset":{"Arguments":{"holdout":9}}}' -n '{}'

python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout0_Netv4","NetworkHPs":{"NetworkConfig":"AE_MNIST_v4.json"},"Dataset":{"Arguments":{"holdout":0}}}' -n '{}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout1_Netv4","NetworkHPs":{"NetworkConfig":"AE_MNIST_v4.json"},"Dataset":{"Arguments":{"holdout":1}}}' -n '{}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout5_Netv4","NetworkHPs":{"NetworkConfig":"AE_MNIST_v4.json"},"Dataset":{"Arguments":{"holdout":5}}}' -n '{}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout7_Netv4","NetworkHPs":{"NetworkConfig":"AE_MNIST_v4.json"},"Dataset":{"Arguments":{"holdout":7}}}' -n '{}'
python Training.py -f MNIST_AE_AD_v1.json -c '{"RunName":"MNIST_AE_AD_holdout9_Netv4","NetworkHPs":{"NetworkConfig":"AE_MNIST_v4.json"},"Dataset":{"Arguments":{"holdout":9}}}' -n '{}'
