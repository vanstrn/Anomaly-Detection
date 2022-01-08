#!/bin/bash

python Training.py -f Anogan.json -c '{"RunName":"AnoGAN_holdout_0_l8","NetworkHPs":{"LatentSize":8},"Dataset":{"Arguments":{"holdout":0}}}'
python Training.py -f Anogan.json -c '{"RunName":"AnoGAN_holdout_1_l8","NetworkHPs":{"LatentSize":8},"Dataset":{"Arguments":{"holdout":1}}}'
python Training.py -f Anogan.json -c '{"RunName":"AnoGAN_holdout_2_l8","NetworkHPs":{"LatentSize":8},"Dataset":{"Arguments":{"holdout":2}}}'
python Training.py -f Anogan.json -c '{"RunName":"AnoGAN_holdout_3_l8","NetworkHPs":{"LatentSize":8},"Dataset":{"Arguments":{"holdout":3}}}'
python Training.py -f Anogan.json -c '{"RunName":"AnoGAN_holdout_4_l8","NetworkHPs":{"LatentSize":8},"Dataset":{"Arguments":{"holdout":4}}}'
python Training.py -f Anogan.json -c '{"RunName":"AnoGAN_holdout_5_l8","NetworkHPs":{"LatentSize":8},"Dataset":{"Arguments":{"holdout":5}}}'
python Training.py -f Anogan.json -c '{"RunName":"AnoGAN_holdout_6_l8","NetworkHPs":{"LatentSize":8},"Dataset":{"Arguments":{"holdout":6}}}'
python Training.py -f Anogan.json -c '{"RunName":"AnoGAN_holdout_7_l8","NetworkHPs":{"LatentSize":8},"Dataset":{"Arguments":{"holdout":7}}}'
python Training.py -f Anogan.json -c '{"RunName":"AnoGAN_holdout_8_l8","NetworkHPs":{"LatentSize":8},"Dataset":{"Arguments":{"holdout":8}}}'
python Training.py -f Anogan.json -c '{"RunName":"AnoGAN_holdout_9_l8","NetworkHPs":{"LatentSize":8},"Dataset":{"Arguments":{"holdout":9}}}'
