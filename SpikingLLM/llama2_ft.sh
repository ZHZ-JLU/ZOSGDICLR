export HF_ENDPOINT=https://hf-mirror.com

METHOD=zo 
TASK=$1         
STEPS=20000
gpu_id=$2
logging_steps=$3
T=$4
IR=$5
EPS=$6

CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
--model_path meta-llama/Llama-2-7b-hf  \
--model_name Llama-2-7b \
--seed 42 \
--mse_init \
--pre_rotate \
--down_online_had \
--qk_online_had \
--set_prefixed_tokens \
--eval_ppl \
--T $T \
--fast \
--train_as_classification True \
--task_name $TASK \
--trainer $METHOD \
--max_steps $STEPS \
--learning_rate $IR \
--logging_steps $logging_steps \
--output_dir ./log_snn_mezo_llama2-7B_T$T/$METHOD-$TASK-$STEPS-$IR-$EPS \
--train \
--epochs 0 \
--use_float \
--zo_eps $EPS
