export PYTHONPATH=$PYTHONPATH:$PWD

cuda=$1

IFS=',' read -ra parts <<< "$cuda"
num_gpus=${#parts[@]}

echo use $num_gpus gpus
port=`comm -23 <(seq 50075 65535 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
#python3 src/train/train.py
accelerate launch --config-file default_config.yaml \
    --main_process_port $port \
    --num_processes ${num_gpus} \
    --gpu_ids ${cuda} \
    src/train/train.py \
    --model-config src/config/160ms.json \
    --batch-size 16 \
    --max-len 1000 \
    --flow-ratio 0.50 \
    --cfg-ratio 0.1 \
    --cfg-scale 2.0 \
    --p 0.5 \
    --num-workers 8 \
    --feature-list "bn mel xvector" \
    --additional-feature-list "inputs_length prompt" \
    --feature-pad-values "0. -1.0 0." \
    --steps 1 \
    --cfg-strength 2.0 \
    --chunk-size 16 \
    --result-dir "results" \
    --save-per-updates 5000 \
    --reset-lr 1 \
    --learning-rate 5e-5 \
    --epochs 1000 \
    --resumable-with-seed 666 \
    --grad-accumulation-steps 1 \
    --grad-ckpt 0 \
    --exp-name emilia_hq_1w_mrte_dis \
    --dataset-path "emilia_hq_1w_mrte" 

