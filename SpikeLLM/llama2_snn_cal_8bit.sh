export HF_ENDPOINT=https://hf-mirror.com

METHOD=zo 
TASK=$1         
gpu_id=$2
logging_steps=$3
STEPS=$4
IR=$5
cal_lr=$6

CUDA_VISIBLE_DEVICES=$gpu_id python main_cal_8bit.py \
--model meta-llama/Llama-2-7b-hf  \
--epochs 0 --q_output_dir ./log_snn_llama2-7b_4bit_cal_8-4bit_cal_$cal_lr/$METHOD-$TASK-$STEPS-$IR \
--wbits 16 --abits 8 --lwc --let \
--let_lr 1e-3 --alpha 0.75 --seed 42 --addbit 3 --low_p 0.95 \
--train_as_classification True \
--task_name $TASK \
--trainer $METHOD \
--max_steps $STEPS \
--learning_rate $IR \
--logging_steps $logging_steps \
--output_dir ./log_snn_llama2-7b_4bit_cal_8-4bit_cal_$cal_lr/$METHOD-$TASK-$STEPS-$IR \
--attn_implementation eager \
--resume ./log_snn_llama2-7b_8bit/omni_parameters.pth \
--cal_epochs 20 \
--cal_save_path ./log_snn_mezo_llama2-7b_8bit/zo-$TASK-$STEPS-1e-7/checkpoint-$STEPS \
--cal_lr $cal_lr \
--use_float \
--cal_seq_len 1024