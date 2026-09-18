## SpikeLLM

# Install
```
conda env create -f environment.yml -n SpikeLLM
conda activate SpikeLLM
```

# Run
```
# Conversion
python generate_act_scale_shift.py
bash llama2_snn_42.sh SST2 0
# Task Learning
bash llama2_snn_ft.sh.sh SST2 0 125 20000
# Knowledge Transfer
bash llama2_snn_cal_8bit.sh SST2 0 125 20000 1e-7 5e-6
```
