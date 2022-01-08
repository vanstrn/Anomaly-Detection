#!/bin/bash

python Training.py -f BasicVAE.json -c '{"RunName":"VAE_holdout_0_l8","NetworkHPs":{"LatentSize":8},"Dataset":{"Name":"MNIST_Anomaly","Arguments":{"holdout":0}}}'
python Training.py -f BasicVAE.json -c '{"RunName":"VAE_holdout_1_l8","NetworkHPs":{"LatentSize":8},"Dataset":{"Name":"MNIST_Anomaly","Arguments":{"holdout":1}}}'
python Training.py -f BasicVAE.json -c '{"RunName":"VAE_holdout_2_l8","NetworkHPs":{"LatentSize":8},"Dataset":{"Name":"MNIST_Anomaly","Arguments":{"holdout":2}}}'
python Training.py -f BasicVAE.json -c '{"RunName":"VAE_holdout_3_l8","NetworkHPs":{"LatentSize":8},"Dataset":{"Name":"MNIST_Anomaly","Arguments":{"holdout":3}}}'
python Training.py -f BasicVAE.json -c '{"RunName":"VAE_holdout_4_l8","NetworkHPs":{"LatentSize":8},"Dataset":{"Name":"MNIST_Anomaly","Arguments":{"holdout":4}}}'
python Training.py -f BasicVAE.json -c '{"RunName":"VAE_holdout_5_l8","NetworkHPs":{"LatentSize":8},"Dataset":{"Name":"MNIST_Anomaly","Arguments":{"holdout":5}}}'
python Training.py -f BasicVAE.json -c '{"RunName":"VAE_holdout_6_l8","NetworkHPs":{"LatentSize":8},"Dataset":{"Name":"MNIST_Anomaly","Arguments":{"holdout":6}}}'
python Training.py -f BasicVAE.json -c '{"RunName":"VAE_holdout_7_l8","NetworkHPs":{"LatentSize":8},"Dataset":{"Name":"MNIST_Anomaly","Arguments":{"holdout":7}}}'
python Training.py -f BasicVAE.json -c '{"RunName":"VAE_holdout_8_l8","NetworkHPs":{"LatentSize":8},"Dataset":{"Name":"MNIST_Anomaly","Arguments":{"holdout":8}}}'
python Training.py -f BasicVAE.json -c '{"RunName":"VAE_holdout_9_l8","NetworkHPs":{"LatentSize":8},"Dataset":{"Name":"MNIST_Anomaly","Arguments":{"holdout":9}}}'
