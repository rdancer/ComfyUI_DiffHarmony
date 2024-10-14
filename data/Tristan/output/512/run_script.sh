export PYTHONPATH=.:$PYTHONPATH
ACC_CONFIG_FILE="configs/acc_configs/single_default.yaml"
export CUDA_VISIBLE_DEVICES="0,1"
NUM_PROCESSES=2
MASTER_PORT=29500

EVAL_BATCH_SIZE=1
NUM_WORKERS=1

OUTPUT_DIR="data/Tristan/output"
mkdir -p $OUTPUT_DIR
cat "$0" >> $OUTPUT_DIR/run_script.sh

DATA_DIR=./data/Tristan
TEST_FILE=test.jsonl # relative to $DATA_DIR

# Paths or HuggingFace IDs of the models from https://huggingface.co/models; I cannot find the DiffHarmony finetuning of SD Inpainting -- rdancer 2024-10-13
SD_MODEL=stable-diffusion-v1-5/stable-diffusion-inpainting
VAE_MODEL=./checkpoints/base/vae
UNET_MODEL=./checkpoints/base/unet



accelerate launch --config_file $ACC_CONFIG_FILE --num_processes $NUM_PROCESSES --main_process_port $MASTER_PORT \
scripts/inference/main.py \
    --pretrained_model_name_or_path "$SD_MODEL" \
    --pretrained_vae_model_name_or_path "$VAE_MODEL" \
    --pretrained_unet_model_name_or_path "$UNET_MODEL" \
    --dataset_root $DATA_DIR \
	--test_file $TEST_FILE \
    --output_dir $OUTPUT_DIR \
	--seed=0 \
	--resolution=512 \
	--output_resolution=512 \
	--eval_batch_size=$EVAL_BATCH_SIZE \
	--dataloader_num_workers=$NUM_WORKERS \
	--mixed_precision="fp16"

	# --stage2_model_name_or_path ""export PYTHONPATH=.:$PYTHONPATH
ACC_CONFIG_FILE="configs/acc_configs/single_default.yaml"
export CUDA_VISIBLE_DEVICES="0,1"
NUM_PROCESSES=2
MASTER_PORT=29500

EVAL_BATCH_SIZE=1
NUM_WORKERS=1

OUTPUT_DIR="data/Tristan/output"
mkdir -p $OUTPUT_DIR
cat "$0" >> $OUTPUT_DIR/run_script.sh

DATA_DIR=./data/Tristan
TEST_FILE=test.jsonl # relative to $DATA_DIR

# Paths or HuggingFace IDs of the models from https://huggingface.co/models; I cannot find the DiffHarmony finetuning of SD Inpainting -- rdancer 2024-10-13
SD_MODEL=stable-diffusion-v1-5/stable-diffusion-inpainting
VAE_MODEL=./checkpoints/base/vae
UNET_MODEL=./checkpoints/base/unet



accelerate launch --config_file $ACC_CONFIG_FILE --num_processes $NUM_PROCESSES --main_process_port $MASTER_PORT \
scripts/inference/main.py \
    --pretrained_model_name_or_path "$SD_MODEL" \
    --pretrained_vae_model_name_or_path "$VAE_MODEL" \
    --pretrained_unet_model_name_or_path "$UNET_MODEL" \
    --dataset_root $DATA_DIR \
	--test_file $TEST_FILE \
    --output_dir $OUTPUT_DIR \
	--seed=0 \
	--resolution=512 \
	--output_resolution=512 \
	--eval_batch_size=$EVAL_BATCH_SIZE \
	--dataloader_num_workers=$NUM_WORKERS \
	--mixed_precision="fp16"

	# --stage2_model_name_or_path ""