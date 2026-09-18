import os
import sys
import random
import numpy as np
# from models.LMClass import LMClass
import torch
import time
# from datautils import get_loaders
# from lm_eval import evaluator
from pprint import pprint
# from parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
import torch.nn as nn
# from spike_driven_quant.spike_omniquant import spike_omniquant
from tqdm import tqdm
import utils
from pathlib import Path
# from categories import subcategories, categories

# from models.spike_llama_layer import QuantLlamaDecoderLayer
# from models.int_opt_layer import QuantOPTDecoderLayer
# from spike_driven_quant.spike_linear import SpikeQuantLinear
# from spike_driven_quant.utils import set_quant_state, set_noise_state

import pdb

import argparse
from tasks import get_task
from dataclasses import dataclass
from transformers import HfArgumentParser, TrainingArguments,DataCollatorForTokenClassification, AutoConfig, AutoTokenizer, AutoModelForCausalLM
from utils import *
from trainer import ZOTrainer, QZOTrainer, QAZOTrainer, LOZOTrainer, SparseZOTrainer, HizooTrainer, ZOAdamWTrainer
from pzotrainer import PZOTrainer
from torch.utils.data import Dataset
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from metrics import calculate_metric
from accelerate import Accelerator
torch.backends.cudnn.benchmark = True

from pathlib import Path
from phase_utils import train_utils
import phase_utils.rotation_utils as rotation_utils
from quantize.int_linear_real import load_quantized_model
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.hooks import remove_hook_from_module
from phase_utils.data_utils import get_loaders, test_ppl
from phase.phase_util import wrap_to_phase_model, init_out_neuron, init_weight_quantizer, init_input_neuron, register_online_had, set_phase_model_time_step, get_act_stat, change_T
from phase_utils.snn_utils import replicate_past_key_values
import phase_utils.model_utils as model_utils

#chents_train
@dataclass
class OurArguments(TrainingArguments):
    # dataset and sampling strategy
    task_name: str = "SST2"  

    # Number of examples
    num_train: int = 1000  # TRAIN=1000
    num_dev: int = 100  
    num_eval: int = 1000  # EVAL=1000

    # Number of training sets (set to None if not specified)
    num_train_sets: int = None  

    # Model loading
    model_name: str = ""  
    load_float16: bool = True  # --load_float16
    load_bfloat16: bool = False  
    load_int8: bool = False 
    max_length: int = 2048  

    # Calibration
    sfc: bool = False 
    icl_sfc: bool = False  

    # Training
    trainer: str = "zo"#"qzo"  # --trainer qzo
    only_train_option: bool = True  
    train_as_classification: bool = True  
    
    # MeZO
    zo_eps: float = 1e-3  # --zo_eps 1e-3

    # QZO Added: Training arguments
    quant_method: str = ''  # --quant_method omniquant
    should_save: bool = True  
    clip_zo_grad: bool = False  # --clip_zo_grad
    train_unquantized: bool = False  
    max_steps: int = 20000 # --max_steps 20000
    learning_rate: float = 1e-7  # --learning_rate 1e-5
    # Generation
    sampling: bool = False  
    temperature: float = 1.0 
    num_beams: int = 1  
    top_k: int = None  
    top_p: float = 0.95  
    max_new_tokens: int = 50  
    eos_token: str = "\n"  

    # Saving
    save_model: bool = False  
    no_eval: bool = False  
    tag: str = "qazo-ft-20000-16-1e-5-1e-3-0"  # tag= qzo-$MODE-$STEPS-$BS-$LR-$EPS-$SEED
    save_total_limit: int = 1  

    # Linear probing
    linear_probing: bool = False  
    lp_early_stopping: bool = False 
    head_tuning: bool = False  

    # Untie emb/lm_head weights
    untie_emb: bool = False  

    # Display
    verbose: bool = False 

    # Non-diff objective
    non_diff: bool = False  

    # Auto saving when interrupted
    save_on_interrupt: bool = False  

    # Additional parameters from the script
    train_set_seed: int = 42  
    result_file: str = None  
    logging_steps: int = 125  
    evaluation_strategy: str = "steps"  
    save_strategy: str = "steps"  
    lr_scheduler_type: str = "constant" 
    output_dir = "./log/test_use"  


    
    max_memory:  str = "15GiB"
    
    #
    train: bool = False  # enable training
    use_float: bool = False

    # LOZO
    step_interval: int = 100 # $\nu$ in LOZO
    rank_r: int = 4 # rank r in LOZO

    #SparseMezo
    ratio: float = 0.3 
    
    # pesuzo
    hessian_smooth: float = 1e-8 # hessian_smooth in HZO
    perturb_type: str = "Gaussian"
    adjust_lm_head: str = "False"
    logits: str = "False"

    #hizoo
    # warmup_step: int = 5000
    # decay_step: int = 0
    # zo_lr_scheduler_type: str = 'cosine_with_warmup'
    # weight_decay: float = 0
    hessian_smooth_type: str = 'constant1e-10'
    # optim: str = "sgd"
    # momentum: float = 0.9
    # adam_beta1: float = 0.0
    # adam_beta2: float = 0.9999999999
    # adam_epsilon: float = 10000.0

    fisher_smooth: float= 0.99

    #PhaseSNN parameter
    model_path: str = None
    model_name: str = None
    cache_dir: str = "./cache"
    # output_dir: str = "./log/"
    save_quant_dir: str = None
    real_quant: bool = False
    resume_quant: str = None
    wbits: int = 16
    w_group_size: int = -1
    w_asym: bool = False
    input_bits: int = 16
    input_group_size: int = -1
    input_mode: str = 'dynamic'
    input_asym: bool = False
    output_mode: str = 'dynamic'
    output_asym: bool = False
    k_bits: int = 16
    output_bits: int = 16
    v_bits: int = 16
    kv_group_size: int = -1
    k_pre_rope: bool = False
    kv_mode: str = 'dynamic'
    kv_asym: bool = False
    mse_init: bool = False
    asym_mse_init: bool = False
    skip_qk_weight_init: bool = False
    block_qk_weight_init: bool = False
    mse_init_size: int = 8
    fp_mse_init: bool = False
    pre_rotate: bool = False
    rotate_mode: str = 'hadamard'
    down_online_had: bool = False
    qk_online_had: bool = False
    set_prefixed_tokens: bool = False
    outlier_threshold: int = 64
    activation_clipping: bool = False
    quant_lr: float = 5e-5
    weight_lr: float = 5e-6
    min_lr_factor: float = 10
    clip_grad: float = 0.3
    wd: float = 0
    off_load_to_disk: bool = False
    use_fp32: bool = False
    # debug: bool = False
    early_stop: int = 0
    constant_wlr: bool = False
    train_size: int = 512
    val_size: int = 64
    training_seqlen: int = 512
    epochs: int = 0
    calib_dataset: str = "pile"
    batch_size: int = 4
    loss_type: str = "mse"
    training_target: str = "fp_input"
    ppl_seqlen: int = 2048
    seed: int = 2
    eval_ppl: bool = False
    # eval_tasks: str = ""
    eval_batch_size: int = 16
    max_outlier: float = 5
    max_item_index: int = 5
    set_outlier_zero: bool = False
    modified_index: int = 0
    ablate_prefix_number: int = None

    T: int = 8
    neuron_path: str = None
    fast: bool = False

    load_cal: bool = False
    spike_model_path: str = None
    prefix_path: str = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]
    print(args)
    return args

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def result_file_tag(args):
    """
    Get the result file tag
    """
    save_model_name = args.model_name.split("/")[-1]
    sfc_tag = "-sfc" if args.sfc else ""
    icl_sfc_tag = "-icl_sfc" if args.icl_sfc else ""
    sample_eval_tag = "-sampleeval%d" % args.num_eval if args.num_eval is not None else ""
    sample_train_tag = "-ntrain%d" % args.num_train if args.num_train > 0 else ""
    sample_dev_tag = "-ndev%d" % args.num_dev if args.num_dev is not None else ""
    customized_tag = f"-{args.tag}" if len(args.tag) > 0 else ""
    return f"{args.task_name}-{save_model_name}" + sfc_tag + icl_sfc_tag + sample_eval_tag + sample_train_tag + sample_dev_tag + customized_tag
class Framework:
    def __init__(self, args, task, model,tokenizer):
        self.args = args
        self.task = task
        self.model, self.tokenizer = model,tokenizer
    def train(self, train_samples, eval_samples):
        """
        Training function
        """
        # Set tokenizer to left padding (so that all the options are right aligned)
        self.tokenizer.padding_side = "left"

        class HFDataset(Dataset):

            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]


        def _convert(samples):
            """
            Convert samples to HF-compatible dataset
            """
            data = []
            for sample in samples:
                encoded_candidates, option_lens = encode_prompt(
                    self.task, self.task.get_template(), [], sample, self.tokenizer, 
                    max_length=self.args.max_length, generation=self.task.generation, generation_with_gold=True, 
                    max_new_tokens=self.args.max_new_tokens
                )
                if self.task.generation:
                    correct_candidate_id = 0
                elif isinstance(sample.correct_candidate, list):
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
                else:
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate)
                
                if self.args.non_diff:
                    # For non-differentiable objective, there is no teacher forcing thus the 
                    # current answer part is removed
                    encoded_candidates[correct_candidate_id] = encoded_candidates[correct_candidate_id][:-option_lens[correct_candidate_id]]

                if self.args.train_as_classification:
                    # For classification, we provide the label as the correct candidate id
                    data.append([{"input_ids": encoded_candidates[_i], "labels": correct_candidate_id, "option_len": option_lens[_i], "num_options": len(sample.candidates)} for _i in range(len(encoded_candidates))])
                elif self.args.only_train_option:
                    # Otherwise, it is just LM-style teacher forcing
                    if self.args.non_diff:
                        # For non-differentiable objective, we need to provide the gold answer to calculate F1/acc
                        data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id], "gold": sample.correct_candidate})
                    else:
                        data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id], "option_len": option_lens[correct_candidate_id]})
                else:
                    data.append({"input_ids": encoded_candidates[correct_candidate_id], "labels": encoded_candidates[correct_candidate_id]})
            return data

        with count_time("Tokenizing training samples"):
            train_dataset = HFDataset(_convert(train_samples))
            eval_dataset = HFDataset(_convert(eval_samples))
        
        if self.args.only_train_option and not self.args.non_diff:
            # If --only_train_option and not with a non-differentiable objective, we wrap the forward function
            self.model.original_forward = self.model.forward
            if self.args.trainer != "pzo":
                self.model.forward = forward_wrap_with_option_len.__get__(self.model, type(self.model))
            else:
                self.model.forward = forward_wrap_with_option_len_pzo.__get__(self.model,type(self.model))

        if self.args.non_diff:
            collator = NondiffCollator
        else:
            collator = DataCollatorForTokenClassification

        if self.args.use_float:
            self.model.float()

        if self.args.trainer == 'zo': # MeZO
            from accelerate import infer_auto_device_map, dispatch_model
            if self.args.set_prefixed_tokens:
                block_class_name = self.model.model.model.layers[0].__class__.__name__
            else:
                block_class_name = self.model.model.layers[0].__class__.__name__
            device_map = infer_auto_device_map(self.model, max_memory={i: self.args.max_memory for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
            self.model = dispatch_model(self.model, device_map=device_map, skip_keys='past_key_values')
            trainer = ZOTrainer(
                model=self.model, 
                args=self.args,
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8)
            )
        elif self.args.trainer == 'hizoo': # MeZO
            trainer = HizooTrainer(
                model=self.model, 
                args=self.args,
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8)
            )
        elif self.args.trainer == 'zoadamw': # MeZO
            from accelerate import infer_auto_device_map, dispatch_model
            block_class_name = self.model.model.layers[0].__class__.__name__
            device_map = infer_auto_device_map(self.model, max_memory={i: self.args.max_memory for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
            self.model = dispatch_model(self.model, device_map=device_map, skip_keys='past_key_values')
            trainer = ZOAdamWTrainer(
                model=self.model, 
                args=self.args,
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8)
            )
        elif self.args.trainer == 'regular': # Fine-tune (this is not used in our experiments)
            from transformers import Trainer
            from accelerate import infer_auto_device_map, dispatch_model
            block_class_name = self.model.model.layers[0].__class__.__name__
            device_map = infer_auto_device_map(self.model, max_memory={i: self.args.max_memory for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
            self.model = dispatch_model(self.model, device_map=device_map, skip_keys='past_key_values')
            trainer = Trainer(
                model=self.model, 
                args=self.args,
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8)
            )
        # QZO Added: set QZOTrainer
        elif self.args.trainer == 'qzo' and self.args.quant_method != '': 
            assert self.args.quant_method in ['gptq', 'omni','aqlm'] # supported methods
            trainer = QZOTrainer(
                model=self.model, 
                args=self.args,
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8)
            )
        elif self.args.trainer == 'qazo' and self.args.quant_method != '': 
            assert self.args.quant_method in ['gptq', 'omni','aqlm'] # supported methods
            trainer = QAZOTrainer(
                model=self.model, 
                args=self.args,
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8)
            )
        elif self.args.trainer == 'lozo':
            # from LOZOtrainer import LowRankTrainer
            trainer = LOZOTrainer(
                model=self.model.cuda(), 
                args=self.args,
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8),
            )
        elif self.args.trainer == 'szo':
            # from LOZOtrainer import LowRankTrainer
            trainer = SparseZOTrainer(
                model=self.model.cuda(), 
                args=self.args,
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8) if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8),
            )
        elif self.args.trainer == 'pzo':
            trainer = PZOTrainer(
                model=self.model.cuda(), 
                args=self.args,
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer, pad_to_multiple_of=8)if self.args.train_as_classification else collator(self.tokenizer, pad_to_multiple_of=8),)
        else:
            raise NotImplementedError

        if self.args.save_on_interrupt:
            trainer.add_callback(SIGUSR1Callback())

        # Resume training from a last checkpoint
        last_checkpoint = None
        from transformers.trainer_utils import get_last_checkpoint
        if os.path.isdir(self.args.output_dir) and not self.args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.args.output_dir)
        if last_checkpoint is not None and self.args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
        if self.args.resume_from_checkpoint is not None:
            last_checkpoint = self.args.resume_from_checkpoint
        #add
        if self.args.max_steps > 0:
            trainer.train(resume_from_checkpoint=last_checkpoint) 
        else:
            accelerator = Accelerator()
            self.model, _ = accelerator.prepare(self.model, torch.optim.AdamW(self.model.parameters()))

        # Explicitly save the model
        if self.args.save_model:
            logger.warn("Save model..")
            trainer.save_model()
        
        # FSDP compatibility
        self.model = trainer.model 
        
        # Reset the forward function for evaluation
        if self.args.only_train_option and not self.args.non_diff:
            if type(self.model) == FSDP:
                logger.info("This is an FSDP model now. Be careful when assigning back the original forward function")
                self.model._fsdp_wrapped_module.forward = self.model._fsdp_wrapped_module.original_forward
            else:
                self.model.forward = self.model.original_forward
    def forward(self, input_ids, option_len=None, generation=False):
        """
        Given input_ids and the length of the option, return the log-likelihood of each token in the option.
        For generation tasks, return the generated text.
        This function is only for inference
        """
        input_ids = torch.tensor([input_ids]).to(self.model.device)

        if generation:
            args = self.args
            # Autoregressive generation
            outputs = self.model.generate(
                input_ids, do_sample=args.sampling, temperature=args.temperature, 
                num_beams=args.num_beams, top_p=args.top_p, top_k=args.top_k, max_new_tokens=min(args.max_new_tokens, args.max_length - input_ids.size(1)), 
                num_return_sequences=1, eos_token_id=[self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1], self.tokenizer.eos_token_id],
            )
            # For generation, directly return the text output
            output_text = self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
            return output_text
        else:
            with torch.inference_mode():
                self.model.eval()
                logits = self.model(input_ids=input_ids).logits
            labels = input_ids[0, 1:]
            logits = logits[0, :-1] 
            log_probs = F.log_softmax(logits, dim=-1)

            selected_log_probs = log_probs[torch.arange(len(labels)).to(log_probs.device), labels.to(log_probs.device)]
            selected_log_probs = selected_log_probs.cpu().detach()
            # Only return the option (candidate) part
            return selected_log_probs[-option_len:]
    def one_step_pred(self, train_samples, eval_sample, verbose=False):
        """
        Return the prediction on the eval sample. In ICL, use train_samples as demonstrations
        """

        verbose = verbose or self.args.verbose
        if verbose:
            logger.info("========= Example =========")
            logger.info(f"Candidate: {eval_sample.candidates}")
            logger.info(f"Correct candidate: {eval_sample.correct_candidate}")


        # Encode (add prompt and tokenize) the sample; if multiple-choice/classification, encode all candidates (options)
        encoded_candidates, option_lens = encode_prompt(
            self.task, self.task.get_template(), train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length, 
            generation=self.task.generation, max_new_tokens=self.args.max_new_tokens
        )

        # Calibration
        if self.args.sfc or self.args.icl_sfc:
            sfc_encoded_candidates, sfc_option_lens = encode_prompt(self.task, self.task.get_template(), 
                train_samples, eval_sample, self.tokenizer, max_length=self.args.max_length,
                sfc=self.args.sfc, icl_sfc=self.args.icl_sfc, generation=self.task.generation, 
                max_new_tokens=self.args.max_new_tokens
            )

        outputs = []
        if self.task.generation:
            # For generation tasks, return the autoregressively-generated text
            output_text = self.forward(encoded_candidates[0], generation=True)
            if verbose:
                logger.info("=== Prompt ===")
                logger.info(self.tokenizer.decode(encoded_candidates[0]))
                logger.info(f"Output: {output_text}") 
            return Prediction(correct_candidate=eval_sample.correct_candidate, predicted_candidate=output_text)
        else:
            # For classification/multiple-choice, calculate the probabilities of all candidates
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                selected_log_probs = self.forward(encoded_candidate, option_len=option_lens[candidate_id])
                if verbose:
                    if candidate_id == 0:
                        logger.info("=== Candidate %d ===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate))
                    else:
                        logger.info("=== Candidate %d (without context)===" % candidate_id)
                        logger.info(self.tokenizer.decode(encoded_candidate).split(self.task.train_sep)[-1])
                    logger.info(f"Log probabilities of the option tokens: {selected_log_probs}")

                if self.args.sfc or self.args.icl_sfc:
                    sfc_selected_log_probs = self.forward(sfc_encoded_candidates[candidate_id], option_len=sfc_option_lens[candidate_id])
                    if verbose:
                        logger.info("=== Candidate %d (without context) SFC ===" % candidate_id)
                        logger.info(self.tokenizer.decode(sfc_encoded_candidates[candidate_id]).split(self.task.train_sep)[-1])
                        logger.info(f"Log probabilities of the option tokens: {sfc_selected_log_probs}")

                outputs.append({"log_probs": selected_log_probs, "sfc_log_probs": sfc_selected_log_probs if self.args.sfc or self.args.icl_sfc else None})

            if self.args.sfc or self.args.icl_sfc:
                # Calibrated probabilities (surface form competition; https://arxiv.org/pdf/2104.08315.pdf)
                # log p(candidate | input) = log p_lm(candidate | input) - log p_lm(candidate | sfc prompt)
                scores = [x['log_probs'].sum().item() - x['sfc_log_probs'].sum().item() for x in outputs]
            else:
                # (Default) length-normalized log probabilities
                # log p(candidate | input) = log p_lm(candidate | input) / |candidate #tokens|
                scores = [x['log_probs'].mean().item() for x in outputs]


            if verbose:
                logger.info(f"Prediction scores: {scores}")

            if isinstance(eval_sample.correct_candidate, list):
                # For some datasets there are multiple correct answers
                correct_candidate_id = [eval_sample.candidates.index(c) for c in eval_sample.correct_candidate]
            else:
                correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)

            return Prediction(correct_candidate=correct_candidate_id, predicted_candidate=int(np.argmax(scores)))


    def evaluate(self, train_samples, eval_samples, one_train_set_per_eval_sample=False):
        """
        Evaluate function. If one_train_set_per_eval_sample is True, then each eval sample has its own training (demonstration) set.
        """
        if one_train_set_per_eval_sample:
            logger.info(f"There are {len(eval_samples)} validation samples and one train set per eval sample")
        else:
            logger.info(f"There are {len(train_samples)} training samples and {len(eval_samples)} validation samples")

        # Prediction loop
        predictions = []  
        for eval_id, eval_sample in enumerate(tqdm(eval_samples)):
            predictions.append(
                self.one_step_pred(train_samples[eval_id] if one_train_set_per_eval_sample else train_samples, eval_sample, verbose=(eval_id < 3))
            )

        # Calculate metrics 
        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}
        return metrics

def main():
    args = parse_args()
    # args.eval_tasks = args.tasks
    args.tag = f"{args.trainer}-{args.max_steps}-{args.per_device_train_batch_size}-{args.learning_rate}-{args.zo_eps}-{args.seed}-{int(time.time())}"  # tag= qzo-$MODE-$STEPS-$BS-$LR-$EPS-$SEED
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_quant_dir:
        Path(args.save_quant_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = train_utils.create_logger(output_dir)
    logger.info(args)
    if args.model_name is None:
        args.model_name = args.model_path.split('/')[-1]
        logger.info(f"model_name is None, setting as {args.model_name}")
    if args.resume_quant:
        # directly load quantized model for evaluation
        model, tokenizer = load_quantized_model(args.resume_quant, args.wbits, args.group_size)
    else:
        # load fp quantized model
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, legacy=False, trust_remote_code=True)
        dtype = torch.float16 if not args.use_fp32 else torch.float32
        model = AutoModelForCausalLM.from_pretrained(args.model_path, config=config, device_map='cpu', torch_dtype=dtype, trust_remote_code=True)
        if args.pre_rotate:
            rotation_utils.fuse_layer_norms(model)
            rotation_utils.rotate_model(model, rotate_mode=args.rotate_mode, online=args.down_online_had)
            model.half()
        for param in model.parameters():
            param.requires_grad = False
        print(model)
        wrap_to_phase_model(model, T=1)
        print(model)

        # register on-line hadadamrd transformation
        if args.pre_rotate and args.down_online_had:
            register_online_had(model)
        # wrap rope for online_had and rope output capture
        # rope_function_name = model_utils.get_rope_function_name(model)
        # layers = model_utils.get_layers(model)
        # for layer in layers:
        #     rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
        #                 layer.self_attn, 
        #                 rope_function_name, 
        #                 config=model.config,
        #                 online_had=args.qk_online_had)

        if args.use_float:
            model.float()

        prefixed_tokens = None                
        prefixed_key_values = None
        args.prefixed_length = 0
        activation_stat = None  
        # include_static = (args.input_mode == "static" and args.input_bits < 16 ) or (args.kv_mode == "static" and (args.k_bits < 16 or args.v_bits < 16))
        if not args.load_cal:
            from phase_utils.stat_utils import get_prefixed_tokens
            # model and data prepaer
            if model.device.type == 'cpu':
                original_device = 'cpu'
                block_class_name = model.model.layers[0].__class__.__name__
                device_map = infer_auto_device_map(model, max_memory={i: args.max_memory for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
                model = dispatch_model(model, device_map=device_map)
            else:
                original_device = 'cuda'
            cal_dataloader, _ = get_loaders(
            args.calib_dataset,
            tokenizer,
            train_size=64,
            val_size=0,
            seed=args.seed,
            seqlen=512,
            )
            # get prefixed tokens
            if args.set_prefixed_tokens:
                tick = time.time()
                prefixed_tokens = get_prefixed_tokens(cal_dataloader, model, tokenizer, args.model_name, outlier_threshold=args.outlier_threshold, activation_type='down_proj')
                logger.info(f"get {len(prefixed_tokens)} prefixed tokens; token id:{prefixed_tokens}; text: {tokenizer.decode(prefixed_tokens)}")
                logger.info(f"time to get prefixed token:{time.time()-tick:.0}s")
                model.config.prefixed_tokens = prefixed_tokens
                args.prefixed_length = len(prefixed_tokens)
                use_cache = model.config.use_cache
                model.config.use_cache = True
                if args.ablate_prefix_number is not None:
                    prefixed_tokens = prefixed_tokens[:args.ablate_prefix_number]
                    logger.info(f'ablation:set prefix as {prefixed_tokens}')
                output = model(torch.tensor([prefixed_tokens], device=model.device), return_dict=True)
                prefixed_key_values = output.past_key_values
                model.config.use_cache = use_cache
                
                # get activation statistic for activation quantization
                #if include_static:
                    #assert args.input_mode == "static" or args.kv_mode == "static","mse_init require static quantization"

                activation_stat = get_act_stat(model, cal_dataloader, 'max', prefixed_tokens, args.down_online_had)
            else:
                activation_stat = get_act_stat(model, cal_dataloader, 'max', None, args.down_online_had)
            if original_device == 'cpu':
                remove_hook_from_module(model, recurse=True)
                # model = model.cpu()
        else:
            prefixed_key_values = torch.load(os.path.join(args.prefix_path, 'prefixed_key_values.pth'), weights_only=False, map_location='cuda:0')
            activation_stat = None

        set_phase_model_time_step(model, args.T, logger, args.fast)

        # init weight quantizer
        # if args.wbits < 16:
        #     logger.info('init weight quantizer')
        #     init_weight_quantizer(args, model, logger)

        if args.neuron_path:
            logger.info('load neuron_parameter')
            neuron_parameter = torch.load(args.neuron_path)
        else:
            neuron_parameter = None
            
        # init input quantizer
        logger.info('init input neuron')
        init_input_neuron(args, model, activation_stat, logger, neuron_parameter)

        logger.info('init output neuron')
        init_out_neuron(args, model, activation_stat, logger, neuron_parameter)

        print(model)

        # if args.load_cal:
        #     from accelerate import load_checkpoint_in_model
        #     block_class_name = model.model.layers[0].__class__.__name__
        #     device_map = infer_auto_device_map(model, max_memory={i: args.max_memory for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
        #     load_checkpoint_in_model(model,checkpoint=args.spike_model_path,device_map=device_map,dtype=torch.float32)

        if args.fast and args.set_prefixed_tokens:
            spike_prefixed_key_values = replicate_past_key_values(prefixed_key_values, 1)
        elif args.set_prefixed_tokens:
            spike_prefixed_key_values = replicate_past_key_values(prefixed_key_values, args.T)
        else:
            spike_prefixed_key_values = None

        train_utils.cleanup_memory()

        spike_prefixed_key_values = model_utils.mv_kv_cache(spike_prefixed_key_values, model)


        if spike_prefixed_key_values is not None:
            model = model_utils.WrappedPrefixCausalLM(model, spike_prefixed_key_values)
        
        if args.load_cal:
            from accelerate import load_checkpoint_in_model
            block_class_name = model.model.model.layers[0].__class__.__name__
            device_map = infer_auto_device_map(model, max_memory={i: args.max_memory for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
            load_checkpoint_in_model(model,checkpoint=args.spike_model_path,device_map=device_map,dtype=torch.float32)

        if args.epochs > 0:
            from phase.phase_calibration import calibration_T8
            # from utils.snn_utils import get_spike_config
            logger.info("=== start quantization Training ===")
            tick = time.time()  

            spike_prefixed_key_values = tuple(
                (k.detach(), v.detach()) for k, v in spike_prefixed_key_values
            )

            cache_trainloader = f'{args.cache_dir}/dataloader_{args.model_name}_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}_train.cache'
            cache_valloader = f'{args.cache_dir}/dataloader_{args.model_name}_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}_val.cache'
            if os.path.exists(cache_trainloader) and os.path.exists(cache_valloader):
                trainloader = torch.load(cache_trainloader)
                logger.info(f"load trainloader from {cache_trainloader}")
                valloader = torch.load(cache_valloader)
                logger.info(f"load valloader from {cache_valloader}")
            else:
                trainloader, valloader = get_loaders(
                    args.calib_dataset,
                    tokenizer,
                    args.train_size,
                    args.val_size,
                    seed=args.seed,
                    seqlen=args.training_seqlen,
                )
                torch.save(trainloader, cache_trainloader)
                torch.save(valloader, cache_valloader)

            calibration_T8(model.model, spike_prefixed_key_values, spike_prefixed_key_values, args, trainloader, valloader, logger)
            # change_T(model.model, 6)
            if args.save_quant_dir:
                logger.info("start saving model")
                model.model.save_pretrained(args.save_quant_dir, safe_serialization=False)
                tokenizer.save_pretrained(args.save_quant_dir)
                torch.save(prefixed_key_values, os.path.join(args.save_quant_dir, 'prefixed_key_values.pth'))
                # quant_config = get_spike_config(args)
                # quant_config['prefixed_tokens'] = prefixed_tokens
                # train_utils.save_dict_as_json(quant_config, os.path.join(args.save_quant_dir, 'prefixequant_config.json'))
                logger.info(f"save model to {args.save_quant_dir} success")

            logger.info(time.time() - tick)
    
    for param in model.parameters():
        param.requires_grad = True
    for name, param in model.named_parameters():
        if name.find('.tau') > -1 or name.find('.v0') > -1 or name.find('.theta') > -1 or name.find('quantizer.B') > -1 or name.find('quantizer.h') > -1 or name.find('oriebd.weight') > -1 :
            param.requires_grad = False
        # print(f"{name}: {param.requires_grad}")
    #chents_add
    if 'opt' in args.model_path.lower():
        tokenizer.bos_token_id = 0
    if "llama" in args.model_path.lower():
        tokenizer.pad_token_id = 0
    #chents_train
    #train_args = parse_args()
    set_seed(args.seed)
    task = get_task(args.task_name)
    train_sets = task.sample_train_sets(num_train=args.num_train, num_dev=args.num_dev, num_eval=args.num_eval, num_train_sets=args.num_train_sets, seed=args.train_set_seed)
    #evaluate(lm, args,logger)
    # def enable_quant(model, wq=True):
    #     set_quant_state(model, weight_quant=False, act_quant=True) 
    #     # for m in model.modules():
    #     #     if isinstance(m, (QuantLinear, QuantMatMul)):
    #     #         m.set_quant_state(weight_quant=False, act_quant=False)
    # enable_quant(lm.model, wq=True)  
    framework = Framework(args, task, model, tokenizer)
    if args.train_set_seed is not None or args.num_train_sets is not None:
        # Eval samples share one (or multiple) training set(s)
        for train_set_id, train_samples in enumerate(train_sets):
            train_set_seed = train_set_id if args.train_set_seed is None else args.train_set_seed

            # Sample eval samples
            if args.num_eval is not None:
                eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
            else:
                eval_samples = task.valid_samples

            if args.trainer != "none":
                if args.num_dev is not None:
                    # Dev samples
                    dev_samples = train_samples[-args.num_dev:] 
                    train_samples = train_samples[:-args.num_dev]
                else:
                    dev_samples = None

                # Training
                # framework.model.cuda()
                if args.train:
                    framework.train(train_samples, dev_samples if dev_samples is not None else eval_samples)
                else:
                    if args.set_prefixed_tokens:
                        block_class_name = framework.model.model.model.layers[0].__class__.__name__
                    else:
                        block_class_name = framework.model.model.layers[0].__class__.__name__
                    device_map = infer_auto_device_map(framework.model, max_memory={i: args.max_memory for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
                    framework.model = dispatch_model(framework.model, device_map=device_map, skip_keys='past_key_values')
                    framework.model.device = framework.model.model.device
                model = framework.model  # Update the model after training
                #evaluate(lm, args,logger)
                #exit(0)
                if not args.no_eval:
                    metrics = framework.evaluate([], eval_samples) # No in-context learning if there is training
                    if dev_samples is not None:
                        dev_metrics = framework.evaluate([], dev_samples) 
                        for m in dev_metrics:
                            metrics["dev_" + m] = dev_metrics[m]
            else:
                assert args.num_dev is None
                # Zero-shot / in-context learning
                metrics = framework.evaluate(train_samples, eval_samples)

            if not args.no_eval:
                logger.info("===== Train set %d =====" % train_set_seed)
                logger.info(metrics)
                if args.local_rank <= 0:
                    write_metrics_to_file(metrics, f"result/{args.model_path}/" + result_file_tag(args) + f"-trainset{train_set_id}.json" if args.result_file is None else args.result_file)

    else:
        # For each eval sample, there is a training set. no training is allowed
        # This is for in-context learning (ICL)
        assert args.trainer == "none"
        if args.num_eval is not None:
            eval_samples = task.sample_subset(data_split="valid", seed=0, num=args.num_eval)
        else:
            eval_samples = task.valid_samples

        metrics = framework.evaluate(train_sets, eval_samples, one_train_set_per_eval_sample=True)
        logger.info(metrics)
        if args.local_rank <= 0:
            write_metrics_to_file(metrics, "result/" + result_file_tag(args) + "-onetrainpereval.json" if args.result_file is None else args.result_file)    
    print(args.output_dir)
    #evaluate(lm, args,logger)


if __name__ == "__main__":
    print(sys.argv)
    main()
