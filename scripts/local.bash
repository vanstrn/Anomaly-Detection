#!/bin/bash
# Created By  : Neale Van Stralen
# version ='1.0'
# ---------------------------------------------------------------------------
###
#   Bash script wich can run through a test list and run all of the experiments in parallel on different GPUs. EX:
#       bash scripts/local.bash scripts/AutoEncoder_AD_v1.bash 0,1 2 kwargs
#   This will run all of the experiments listed in 'AutoEncoder_AD_v1.bash' on GPUs 1 & 2,
#   two per GPU (4 Total) at a time, with any additional arguments in kwargs
###
# ---------------------------------------------------------------------------

all_args=("$@")
FILE=$1
GPUString=$2
PerGPU=$(($3))
Postfix_=("${all_args[@]:3}")
Postfix="${Postfix_[@]}"

IFS=',' read -ra GPUList <<< "$GPUString"

NumGPU=${#GPUList[@]}

if [ -z "$FILE" ]
then
      echo "No file given. Skipping."
      exit 1
fi

BatchSize=$((PerGPU*NumGPU))

i1=1
i2=0
j=0
while IFS= read -r line
do
  if [ -z "$line" ]; #If the line is empty
    then continue
  fi
  if [[ $line =~ ^#.* ]]; # In the line starts with '#'
    then continue
  fi

  if [ $i2 -eq $PerGPU ]; then
    j=$((j+1))
    i2=0
  fi
  Prefix="CUDA_VISIBLE_DEVICES=${GPUList[$j]}"


  if [ $i1 -lt $BatchSize ]; then
    eval "${Prefix} ${line} ${Postfix}&"
    i1=$((i1+1))
    i2=$((i2+1))
  else
    eval "${Prefix} ${line} ${Postfix} &"
    wait
    i1=1
    i2=0
    j=0
  fi
done < "$FILE"
wait
