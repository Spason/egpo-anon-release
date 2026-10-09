timestamp="GRPO_sparse_reward"
pg_name="grpo"

# Wandb
export WANDB_PROJECT="egpo"

# Support passing DATA_PATH from outside, use default if not provided
if [ -z "${DATA_PATH}" ]; then
    DATA_PATH='scienceqa_data_with_thinking2'
fi

CKPT_PATH="<YOUR_MODEL_PATH>"  # e.g., path to Qwen3-VL-8B-Instruct

llm=${CKPT_PATH}
temp_url="<YOUR_TEMP_MODEL_PATH>"  # Temporary path for model copy

# Refuse to run with unedited placeholders
case "${CKPT_PATH}${temp_url}" in
    *"<YOUR_"*) echo "Please set CKPT_PATH and temp_url in $(basename "${BASH_SOURCE[0]:-$0}") first." >&2; return 1 2>/dev/null || exit 1 ;;
esac

# Model file copy — only rank 0 executes, other nodes wait
DONE_FILE="${temp_url}.done"

if [ "${INDEX}" == "0" ]; then
    rm -f "${DONE_FILE}"
    if [ -d ${temp_url} ]; then
        rm -rf ${temp_url}
    fi
    cp -rf ${llm} ${temp_url}
    touch "${DONE_FILE}"
    echo "Node 0: Model file copy completed"
else
    echo "Node ${INDEX}: Waiting for master node to finish file copy..."
    sleep 5
    while [ ! -f "${DONE_FILE}" ]; do
        sleep 2
    done
    echo "Node ${INDEX}: File copy completed, continuing execution"
fi

RUN_NAME=${DATA_PATH}_${CKPT_PATH##*/}_${timestamp}
SAVE_PATH="./output/${pg_name}/${RUN_NAME}"
CKPT_PATH=${temp_url}
export DEBUG_MODE="true"
export LOG_PATH="./${SAVE_PATH}/debug_log.txt"

wandb login ${WANDB_API_KEY}  # Set WANDB_API_KEY environment variable before running
mkdir -p ${SAVE_PATH}
split_ratio=0.02
SPLIT_DONE_FILE="${SAVE_PATH}/.split_done"
if [ "${INDEX}" == "0" ]; then
    rm -f "${SPLIT_DONE_FILE}"
    python3 src/virft/src/open_r1/split_data.py --dataset_name ${DATA_PATH} --split_ratio ${split_ratio}
    touch "${SPLIT_DONE_FILE}"
    echo "Node 0: Data split completed"
else
    echo "Node ${INDEX}: Waiting for master node to finish data split..."
    sleep 5
    while [ ! -f "${SPLIT_DONE_FILE}" ]; do
        sleep 2
    done
    echo "Node ${INDEX}: Data split completed, continuing execution"
fi

export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_FUSED_ADAM=1

# Launch training
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=${HOST_GPU_NUM} \
         --master_addr=${CHIEF_IP} \
         --master_port=${MASTER_PORT} \
         --nnodes ${HOST_NUM} \
         --node_rank=${INDEX} \
    src/virft/src/open_r1/grpo_egpo.py \
    --output_dir ${SAVE_PATH}  \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --deepspeed src/virft/local_scripts/zero3_qwenvl.json \
    --pg_name ${pg_name} \
    --temperature 0.99 \
    --max_prompt_length 30000 \
    --max_completion_length 1500 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 40140800 \
    --num_train_epochs ${NUM_TRAIN_EPOCHS:-2} \
    --run_name "${RUN_NAME}" \
    --save_steps 50 \
    --eval_steps 20 \
    --eval_strategy "steps" \
    --save_only_model true \
    --num_generations 8 \
    --supervise_thinking_ratio 0.7 \
    --system_prompt_lang "en" \
    --system_prompt_prob 100 \
    --reward_mode sparse \
    --beta 0.04 \
    --epsilon 0.2 \
    --learning_rate 5e-7 \
    2>&1 | tee -a "./${SAVE_PATH}/training_log.log"
