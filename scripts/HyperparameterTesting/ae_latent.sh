\#!/bin/bash

#Default arguments
gpu=""
args=""
seed=""
hyperparams=""
batchID=""

while getopts g:a:s:h:i: flag
do
    case "${flag}" in
        g) gpu=${OPTARG};;
        a) args=${OPTARG};;
        s) seed=${OPTARG};;
        h) hyperparams=${OPTARG};;
        i) batchID=${OPTARG};;
    esac
done


#Createing proper GPU strings.
CUDA_STRING=""
CPU_STRING=""
case "$gpu" in

  "cpu")
    CPU_STRING="-p cpu";
    ;;

  [0-256])
    CUDA_STRING="export CUDA_VISIBLE_DEVICES=$gpu;"
    ;;
  *) ;;
esac

case "$seed" in

  [0-999999])
    SEED_STRING="seed $seed;"
    ;;
  *)   SEED_STRING="";;

esac

$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_1_L2'$batchID'","NetworkHPs":{"LatentSize":2'$hyperparams'},"Dataset":{"Arguments":{"holdout":1}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_1_L4'$batchID'","NetworkHPs":{"LatentSize":4'$hyperparams'},"Dataset":{"Arguments":{"holdout":1}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_1_L6'$batchID'","NetworkHPs":{"LatentSize":6'$hyperparams'},"Dataset":{"Arguments":{"holdout":1}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_1_L8'$batchID'","NetworkHPs":{"LatentSize":8'$hyperparams'},"Dataset":{"Arguments":{"holdout":1}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_1_L10'$batchID'","NetworkHPs":{"LatentSize":10'$hyperparams'},"Dataset":{"Arguments":{"holdout":1}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_1_L12'$batchID'","NetworkHPs":{"LatentSize":12'$hyperparams'},"Dataset":{"Arguments":{"holdout":1}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_1_L14'$batchID'","NetworkHPs":{"LatentSize":14'$hyperparams'},"Dataset":{"Arguments":{"holdout":1}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_1_L16'$batchID'","NetworkHPs":{"LatentSize":16'$hyperparams'},"Dataset":{"Arguments":{"holdout":1}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_1_L18'$batchID'","NetworkHPs":{"LatentSize":18'$hyperparams'},"Dataset":{"Arguments":{"holdout":1}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_1_L20'$batchID'","NetworkHPs":{"LatentSize":20'$hyperparams'},"Dataset":{"Arguments":{"holdout":1}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_1_L30'$batchID'","NetworkHPs":{"LatentSize":30'$hyperparams'},"Dataset":{"Arguments":{"holdout":1}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_1_L40'$batchID'","NetworkHPs":{"LatentSize":40'$hyperparams'},"Dataset":{"Arguments":{"holdout":1}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_1_L50'$batchID'","NetworkHPs":{"LatentSize":50'$hyperparams'},"Dataset":{"Arguments":{"holdout":1}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_1_L75'$batchID'","NetworkHPs":{"LatentSize":75'$hyperparams'},"Dataset":{"Arguments":{"holdout":1}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_1_L100'$batchID'","NetworkHPs":{"LatentSize":100'$hyperparams'},"Dataset":{"Arguments":{"holdout":1}}}' $SEED_STRING $args $CPU_STRING

$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_3_L2'$batchID'","NetworkHPs":{"LatentSize":2'$hyperparams'},"Dataset":{"Arguments":{"holdout":3}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_3_L4'$batchID'","NetworkHPs":{"LatentSize":4'$hyperparams'},"Dataset":{"Arguments":{"holdout":3}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_3_L6'$batchID'","NetworkHPs":{"LatentSize":6'$hyperparams'},"Dataset":{"Arguments":{"holdout":3}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_3_L8'$batchID'","NetworkHPs":{"LatentSize":8'$hyperparams'},"Dataset":{"Arguments":{"holdout":3}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_3_L10'$batchID'","NetworkHPs":{"LatentSize":10'$hyperparams'},"Dataset":{"Arguments":{"holdout":3}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_3_L12'$batchID'","NetworkHPs":{"LatentSize":12'$hyperparams'},"Dataset":{"Arguments":{"holdout":3}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_3_L14'$batchID'","NetworkHPs":{"LatentSize":14'$hyperparams'},"Dataset":{"Arguments":{"holdout":3}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_3_L16'$batchID'","NetworkHPs":{"LatentSize":16'$hyperparams'},"Dataset":{"Arguments":{"holdout":3}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_3_L18'$batchID'","NetworkHPs":{"LatentSize":18'$hyperparams'},"Dataset":{"Arguments":{"holdout":3}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_3_L20'$batchID'","NetworkHPs":{"LatentSize":20'$hyperparams'},"Dataset":{"Arguments":{"holdout":3}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_3_L30'$batchID'","NetworkHPs":{"LatentSize":30'$hyperparams'},"Dataset":{"Arguments":{"holdout":3}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_3_L40'$batchID'","NetworkHPs":{"LatentSize":40'$hyperparams'},"Dataset":{"Arguments":{"holdout":3}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_3_L50'$batchID'","NetworkHPs":{"LatentSize":50'$hyperparams'},"Dataset":{"Arguments":{"holdout":3}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_3_L75'$batchID'","NetworkHPs":{"LatentSize":75'$hyperparams'},"Dataset":{"Arguments":{"holdout":3}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_3_L100'$batchID'","NetworkHPs":{"LatentSize":100'$hyperparams'},"Dataset":{"Arguments":{"holdout":3}}}' $SEED_STRING $args $CPU_STRING

$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_5_L2'$batchID'","NetworkHPs":{"LatentSize":2'$hyperparams'},"Dataset":{"Arguments":{"holdout":5}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_5_L4'$batchID'","NetworkHPs":{"LatentSize":4'$hyperparams'},"Dataset":{"Arguments":{"holdout":5}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_5_L6'$batchID'","NetworkHPs":{"LatentSize":6'$hyperparams'},"Dataset":{"Arguments":{"holdout":5}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_5_L8'$batchID'","NetworkHPs":{"LatentSize":8'$hyperparams'},"Dataset":{"Arguments":{"holdout":5}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_5_L10'$batchID'","NetworkHPs":{"LatentSize":10'$hyperparams'},"Dataset":{"Arguments":{"holdout":5}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_5_L12'$batchID'","NetworkHPs":{"LatentSize":12'$hyperparams'},"Dataset":{"Arguments":{"holdout":5}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_5_L14'$batchID'","NetworkHPs":{"LatentSize":14'$hyperparams'},"Dataset":{"Arguments":{"holdout":5}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_5_L16'$batchID'","NetworkHPs":{"LatentSize":16'$hyperparams'},"Dataset":{"Arguments":{"holdout":5}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_5_L18'$batchID'","NetworkHPs":{"LatentSize":18'$hyperparams'},"Dataset":{"Arguments":{"holdout":5}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_5_L20'$batchID'","NetworkHPs":{"LatentSize":20'$hyperparams'},"Dataset":{"Arguments":{"holdout":5}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_5_L30'$batchID'","NetworkHPs":{"LatentSize":30'$hyperparams'},"Dataset":{"Arguments":{"holdout":5}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_5_L40'$batchID'","NetworkHPs":{"LatentSize":40'$hyperparams'},"Dataset":{"Arguments":{"holdout":5}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_5_L50'$batchID'","NetworkHPs":{"LatentSize":50'$hyperparams'},"Dataset":{"Arguments":{"holdout":5}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_5_L75'$batchID'","NetworkHPs":{"LatentSize":75'$hyperparams'},"Dataset":{"Arguments":{"holdout":5}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_5_L100'$batchID'","NetworkHPs":{"LatentSize":100'$hyperparams'},"Dataset":{"Arguments":{"holdout":5}}}' $SEED_STRING $args $CPU_STRING

$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_9_L2'$batchID'","NetworkHPs":{"LatentSize":2'$hyperparams'},"Dataset":{"Arguments":{"holdout":9}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_9_L4'$batchID'","NetworkHPs":{"LatentSize":4'$hyperparams'},"Dataset":{"Arguments":{"holdout":9}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_9_L6'$batchID'","NetworkHPs":{"LatentSize":6'$hyperparams'},"Dataset":{"Arguments":{"holdout":9}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_9_L8'$batchID'","NetworkHPs":{"LatentSize":8'$hyperparams'},"Dataset":{"Arguments":{"holdout":9}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_9_L10'$batchID'","NetworkHPs":{"LatentSize":10'$hyperparams'},"Dataset":{"Arguments":{"holdout":9}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_9_L12'$batchID'","NetworkHPs":{"LatentSize":12'$hyperparams'},"Dataset":{"Arguments":{"holdout":9}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_9_L14'$batchID'","NetworkHPs":{"LatentSize":14'$hyperparams'},"Dataset":{"Arguments":{"holdout":9}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_9_L16'$batchID'","NetworkHPs":{"LatentSize":16'$hyperparams'},"Dataset":{"Arguments":{"holdout":9}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_9_L18'$batchID'","NetworkHPs":{"LatentSize":18'$hyperparams'},"Dataset":{"Arguments":{"holdout":9}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_9_L20'$batchID'","NetworkHPs":{"LatentSize":20'$hyperparams'},"Dataset":{"Arguments":{"holdout":9}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_9_L30'$batchID'","NetworkHPs":{"LatentSize":30'$hyperparams'},"Dataset":{"Arguments":{"holdout":9}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_9_L40'$batchID'","NetworkHPs":{"LatentSize":40'$hyperparams'},"Dataset":{"Arguments":{"holdout":9}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_9_L50'$batchID'","NetworkHPs":{"LatentSize":50'$hyperparams'},"Dataset":{"Arguments":{"holdout":9}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_9_L75'$batchID'","NetworkHPs":{"LatentSize":75'$hyperparams'},"Dataset":{"Arguments":{"holdout":9}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f AE_AD.json -c '{"RunName":"AE_holdout_9_L100'$batchID'","NetworkHPs":{"LatentSize":100'$hyperparams'},"Dataset":{"Arguments":{"holdout":9}}}' $SEED_STRING $args $CPU_STRING
