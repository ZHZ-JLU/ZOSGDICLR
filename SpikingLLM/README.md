## SpikingLLM

# Install
```
conda env create -f environment.yml -n SpikingLLM
conda activate SpikingLLM
git clone git@github.com:Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
pip install -e .
```

# Run
```
# Task Learning
bash llama2_ft.sh SST2 0 125 8 2e-8 1e-4
# Knowledge Transfer
bash llama2_cal.sh SST2 0 125 8 3e-8 1e-4 5e-6
```
