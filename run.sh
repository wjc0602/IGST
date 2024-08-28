#!/bin/bash
# 增加Tokne数量
export HF_ENDPOINT="https://hf-mirror.com"

RESOLUTION=512 # 分辨率
TBS=3 # BatchSize
REPEATS=1
LRU=0.8e-5 # UNet 学习率
LRT=0.4e-5 # TextEncoder学习率
TIF=1.0 # TextEncoder训练的比例
NNT=1  # 添加新的Token的数量
LS="constant" # 
LWS=0 #
MTS=400 # MAX_TRAIN_STEPS
GPU_COUNT=1
MAX_NUM=28

MODEL_NAME="models/stabilityai-stable-diffusion-2-1"
BASE_INSTANCE_DIR="dataset/TrainB"
BASE_CLASS_DIR="dataset/ClassB"
OUTPUT_DIR_PREFIX="checkpoints/01_e400_lru${LRU}_lrt${LRT}/style_"

for ((folder_number = 1; folder_number < 2; folder_number+=$GPU_COUNT)); do
    for ((gpu_id = 0; gpu_id < GPU_COUNT; gpu_id++)); do
        current_folder_number=$((folder_number + gpu_id))
        if [ $current_folder_number -gt $MAX_NUM ]; then
            break
        fi
        INSTANCE_DIR="${BASE_INSTANCE_DIR}/$(printf "%02d" $current_folder_number)/images"
        CLASS_DIR="${BASE_CLASS_DIR}/$(printf "%02d" $current_folder_number)/images"
        OUTPUT_DIR="${OUTPUT_DIR_PREFIX}$(printf "%02d" $current_folder_number)"

        COMMAND="python train.py \
            --pretrained_model_name_or_path=$MODEL_NAME \
            --instance_data_dir=$INSTANCE_DIR \
            --class_data_dir=$CLASS_DIR \
            --output_dir=$OUTPUT_DIR \
            --instance_prompt="TOK" \
            --resolution=$RESOLUTION \
            --train_batch_size=$TBS \
            --gradient_accumulation_steps=1 \
            --learning_rate=$LRU \
            --lr_scheduler=$LS \
            --lr_warmup_steps=$LWS \
            --max_train_steps=$MTS \
            --repeats=$REPEATS \
            --test \
            --num_inference_steps 50 \
            --train_text_encoder_ti \
            --text_encoder_lr=$LRT \
            --train_text_encoder_ti_frac=$TIF \
            --num_new_tokens_per_abstraction=$NNT \
            --save_checkpoint \
            --seed=0"

        eval $COMMAND &
        sleep 3
    done
    wait
done


EXP_NAME="01_e250_lru1.0e-5_lrt0.5e-5"
NIS=50
SEED=1234

python test_2.py --name_exp $EXP_NAME --num_inference_steps $NIS --seed $SEED