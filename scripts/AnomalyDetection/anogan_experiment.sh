#!/bin/bash

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

$CUDA_STRING python Training.py -f Anogan.json -c '{"RunName":"AnoGAN_holdout_0'$batchID'","NetworkHPs":{'$hyperparams'},"Dataset":{"Arguments":{"holdout":0}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f Anogan.json -c '{"RunName":"AnoGAN_holdout_1'$batchID'","NetworkHPs":{'$hyperparams'},"Dataset":{"Arguments":{"holdout":1}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f Anogan.json -c '{"RunName":"AnoGAN_holdout_2'$batchID'","NetworkHPs":{'$hyperparams'},"Dataset":{"Arguments":{"holdout":2}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f Anogan.json -c '{"RunName":"AnoGAN_holdout_3'$batchID'","NetworkHPs":{'$hyperparams'},"Dataset":{"Arguments":{"holdout":3}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f Anogan.json -c '{"RunName":"AnoGAN_holdout_4'$batchID'","NetworkHPs":{'$hyperparams'},"Dataset":{"Arguments":{"holdout":4}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f Anogan.json -c '{"RunName":"AnoGAN_holdout_5'$batchID'","NetworkHPs":{'$hyperparams'},"Dataset":{"Arguments":{"holdout":5}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f Anogan.json -c '{"RunName":"AnoGAN_holdout_6'$batchID'","NetworkHPs":{'$hyperparams'},"Dataset":{"Arguments":{"holdout":6}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f Anogan.json -c '{"RunName":"AnoGAN_holdout_7'$batchID'","NetworkHPs":{'$hyperparams'},"Dataset":{"Arguments":{"holdout":7}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f Anogan.json -c '{"RunName":"AnoGAN_holdout_8'$batchID'","NetworkHPs":{'$hyperparams'},"Dataset":{"Arguments":{"holdout":8}}}' $SEED_STRING $args $CPU_STRING
$CUDA_STRING python Training.py -f Anogan.json -c '{"RunName":"AnoGAN_holdout_9'$batchID'","NetworkHPs":{'$hyperparams'},"Dataset":{"Arguments":{"holdout":9}}}' $SEED_STRING $args $CPU_STRING
