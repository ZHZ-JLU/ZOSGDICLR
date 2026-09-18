export HF_ENDPOINT=https://hf-mirror.com

METHOD=zo 
TASK=$1         
STEPS=20000
IR=1e-7
gpu_id=$2
logging_steps=125

CUDA_VISIBLE_DEVICES=$gpu_id python main_cal2.py \
--model meta-llama/Llama-2-7b-hf  \
--epochs 0 --q_output_dir ./log_snn_llama2-7b_8bit/$METHOD-$TASK-$STEPS-$IR \
--wbits 16 --abits 8 --lwc --let \
--let_lr 1e-3 --alpha 0.75 --seed 42 --addbit 3 --low_p 0.95 \
--train_as_classification True \
--task_name $TASK \
--trainer $METHOD \
--max_steps $STEPS \
--learning_rate $IR \
--logging_steps $logging_steps \
--output_dir ./log_snn_llama2-7b_8bit/$METHOD-$TASK-$STEPS-$IR \
--attn_implementation eager \
--resume ./log_snn_llama2-7b_8bit/omni_parameters.pth \
--deactive_amp