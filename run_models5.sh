#!/bin/bash

input_path=$1

echo 'Preprocessing...'
python ./scripts/image_processing.py -in=$input_path

echo 'RUN twonet x4'
python ./scripts/inference_twonet.py -x=4 -m=twonet_finetune_x4 -mp='./models5' -f=2 -c=1024 -s=512 -n=0
python ./scripts/inference_twonet.py -x=4 -m=twonet_k2_x4 -mp='./models5' -f=2 -c=1024 -s=512 -n=1

echo 'RUN twonet x2'
python ./scripts/inference_twonet.py -x=2 -m=twonet_finetune_x2 -mp='./models5' -f=2 -c=1024 -s=512 -n=0
python ./scripts/inference_twonet.py -x=2 -m=twonet_finetune_k2_x2 -mp='./models5' -f=2 -c=1024 -s=512 -n=1

echo 'RUN twonet x0'
python ./scripts/inference_twonet.py -x=0 -m=twonet_finetune_x0 -mp='./models5' -f=2 -c=2048 -s=1024 -n=0 -t
python ./scripts/inference_twonet.py -x=0 -m=twonet_finetune_k2_x0 -mp='./models5' -f=2 -c=2048 -s=1024 -n=1 -t

echo 'RUN fusionnet x4'
python ./scripts/inference_fusionnet.py -x=4 -m=fusionnet_finetune_x4 -mp='./models5' -f=1 -c=1024 -s=512 -n=0
python ./scripts/inference_fusionnet.py -x=4 -m=fusionnet_k2_x4 -mp='./models5' -f=1 -c=1024 -s=512 -n=1

echo 'RUN fusionnet x2'
python ./scripts/inference_fusionnet.py -x=2 -m=fusionnet_finetune_x2 -mp='./models5' -f=1 -c=1024 -s=512 -n=0
python ./scripts/inference_fusionnet.py -x=2 -m=fusionnet_k2_x2 -mp='./models5' -f=1 -c=1024 -s=512 -n=1

echo 'RUN fusionnet x0'
python ./scripts/inference_fusionnet.py -x=0 -m=fusionnet_finetune_x0 -mp='./models5' -f=2 -c=2048 -s=1024 -n=0 -t
python ./scripts/inference_fusionnet.py -x=0 -m=fusionnet_finetune_k2_x0  -mp='./models5' -f=2 -c=2048 -s=1024 -n=1 -t

echo 'FUSION'
python ./scripts/ensemble2.py -in=$input_path

rm -r ./caches
rm -r ./test_x0
rm -r ./test_x2
rm -r ./test_x4

zip -r submission.zip ./submission
