import os
import sys
import random
import numpy as np
from models.LMClass import LMClass
import torch
import time
from datautils import get_loaders
from lm_eval import evaluator
from pprint import pprint
from parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
import torch.nn as nn
from spike_driven_quant.spike_omniquant import spike_omniquant, spike_cal, spike_cal_8bit, spike_omniquant_opt
from tqdm import tqdm
import utils
from pathlib import Path
from categories import subcategories, categories

from models.spike_llama_layer import QuantLlamaDecoderLayer
from models.spike_opt_layer import QuantOPTDecoderLayer
from spike_driven_quant.spike_linear import SpikeQuantLinear
from spike_driven_quant.utils import set_quant_state, set_noise_state, set_bit_state

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

net_choices = [
    "opt-125m",
    "opt-1.3b",
    "opt-2.7b",
    "opt-6.7b",
    "opt-13b",
    "opt-30b",
    "opt-66b",
    "llama-7b",
    "llama-13b",
    "llama-30b",
    "llama-65b",
    "Llama-2-7b",
    "Llama-2-13b",
    "Llama-2-70b",
    "Llama-2-7b-chat",
    "Llama-2-13b-chat",
    "llava-llama-2-13b-chat-lightning-preview",
    "falcon-180b",
    "falcon-7b",
    "mixtral-8x7b"
]

#chents_train
@dataclass
class OurArguments(TrainingArguments):
    # dataset and sampling strategy
    task_name: str = "SST2" 

    # Number of examples
    num_train: int = 1000 
    num_dev: int = 100  
    num_eval: int = 1000 

    # Number of training sets (set to None if not specified)
    num_train_sets: int = None  

    # Model loading
    model_name: str = ""  
    load_float16: bool = True  
    load_bfloat16: bool = False  
    load_int8: bool = False  
    max_length: int = 2048  

    # Calibration
    sfc: bool = False  
    icl_sfc: bool = False 

    # Training
    trainer: str = "zo"#"qzo"  
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

    model_path: str = None
     # 
    model: str = "meta-llama/Llama-2-7B-hf"  # model name or path
    cache_dir: str = "./cache"  # cache dir of dataset
    q_output_dir: str = "../log/"  # direction of logging file
    save_dir: Optional[str] = None  # direction for saving fake quantization model
    resume: Optional[str] = None  # resume training
    real_quant: bool = False  # real quantization for memory reduction
    
    # 
    calib_dataset: str = "wikitext2"  # calibration dataset
    nsamples: int = 128  # Number of calibration data samples
    batch_size: int = 1  # batch size for calibration
    wbits: int = 4  # weight bits
    abits: int = 16  # activation bits
    group_size: Optional[int] = None  # group size for quantization
    alpha: float = 0.5  # alpha parameter
    let_lr: float = 5e-3  # learnable equivalent transformation learning rate
    lwc_lr: float = 1e-2  # learnable weight clipping learning rate
    wd: float = 0.0  # weight decay
    epochs: int = 10  # number of epochs
    let: bool = False  # activate learnable equivalent transformation
    lwc: bool = False  # activate learnable weight clipping
    aug_loss: bool = False  # calculate additional loss with same input
    symmetric: bool = False  # symmetric quantization
    disable_zero_point: bool = False  # quantization without zero_point
    a_dynamic_method: str = "per_token"  # activation dynamic method
    w_dynamic_method: str = "per_channel"  # weight dynamic method

    #SpikeLLM
    addbit: int = 1
    low_p: float = 1.0

    
    # 
    tasks: str = ""  # evaluation tasks
    eval_ppl: bool = False  # evaluate perplexity
    num_fewshot: int = 0  # number of few-shot examples
    max_memory:  str = "80GiB"
    
    # 
    limit: int = -1  # limit number of samples
    multigpu: bool = False  # map model to multiple gpus at eval
    deactive_amp: bool = False  # deactivate AMP when 8<=bits<16
    attn_implementation: str = "sdpa"  # attention implementation  ####################################
    net: Optional[str] = None  # network choice
    act_scales: Optional[str] = None  # activation scales file
    act_shifts: Optional[str] = None  # activation shifts file
    train: bool = False  # enable training
    use_float: bool = False

    #cal
    cal_epochs: int = 0
    cal_save_path: str = None
    load_cal: bool = False
    cal_lr: float = 1e-6
    cal_nsamples: int = 512
    cal_seq_len: int = 512
    cal_batch_size: int = 4

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
    args.eval_tasks = args.tasks
    args.tag = f"{args.trainer}-{args.max_steps}-{args.per_device_train_batch_size}-{args.learning_rate}-{args.zo_eps}-{args.seed}-{int(time.time())}"  # tag= qzo-$MODE-$STEPS-$BS-$LR-$EPS-$SEED
    initial_seed = 2
    
    random.seed(initial_seed)
    np.random.seed(initial_seed)
    torch.manual_seed(initial_seed)
    torch.cuda.manual_seed(initial_seed)

    # check
    if args.epochs > 0:
        assert args.lwc or args.let
        
    if (args.wbits<16 and args.wbits>=8) or (args.abits<16 and args.abits>=8):
        args.deactive_amp = True

    # init logger
    if args.q_output_dir:
        Path(args.q_output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    q_output_dir = Path(args.q_output_dir)
    logger = utils.create_logger(q_output_dir)
    logger.info(args)
    
    # load model
    if args.net is None:
        args.net = args.model.split('/')[-1]
    # assert args.net in net_choices
    args.model_family = args.net.split('-')[0]
    lm = LMClass(args)
    lm.seqlen = 2048
    lm.model.eval()
    # for param in lm.model.parameters():
    #     param.requires_grad = False

    

    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "lwc":args.lwc,
        "disable_zero_point": args.disable_zero_point
    }
    args.act_quant_params = {
        "n_bits":  args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
        "addbit": args.addbit,
        "low_p": args.low_p,
    }
    args.q_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
        "addbit": args.addbit,
        "low_p": args.low_p,
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
        "addbit": args.addbit,
        "low_p": args.low_p,
    }
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }

    if args.multigpu:
        gpu_id = get_lowest_occupied_gpu(wait_memory=5000)
        lm._device = f"cuda:{gpu_id}"
        logger.info(f"set quantization in gpu {gpu_id}")

    # act scales and shifts
    if args.act_scales is None:
        args.act_scales = f'./act_scales/{args.net}.pt'
    if args.act_shifts is None:
        args.act_shifts = f'./act_shifts/{args.net}.pt'

    # quantization
    if args.wbits < 16 or args.abits <16:
        logger.info("=== start quantization ===")
        tick = time.time()     
        # load calibration dataset
        cache_dataloader = f'{args.cache_dir}/dataloader_{args.model_family}_{args.calib_dataset}_{args.nsamples}.cache'
        if os.path.exists(cache_dataloader):
            dataloader = torch.load(cache_dataloader)
            logger.info(f"load calibration from {cache_dataloader}")
        else:
            dataloader, _ = get_loaders(
                args.calib_dataset,
                nsamples=args.nsamples,
                seed=initial_seed,
                model=args.model,
                seqlen=lm.seqlen,
            )
            torch.save(dataloader, cache_dataloader)    
        act_scales = None
        act_shifts = None
        if args.let:
            #act_scales = torch.load(args.act_scales)
            #act_shifts = torch.load(args.act_shifts)
            print("load act scales and shifts from ", args.act_scales, args.act_shifts)
            act_scales = torch.load(args.act_scales,weights_only=False)
            act_shifts = torch.load(args.act_shifts,weights_only=False)
        spike_omniquant(
            lm,
            args,
            dataloader,
            act_scales,
            act_shifts,
            logger,
        )
        if args.cal_epochs > 0:
            cal_dataloader, _ = get_loaders(
                args.calib_dataset,
                nsamples=args.cal_nsamples,
                seed=initial_seed,
                model=args.model,
                seqlen=args.cal_seq_len,
            )
            from accelerate import load_checkpoint_in_model, infer_auto_device_map
            if "llama" in args.net.lower():
                block_class_name = lm.model.model.layers[0].__class__.__name__
            elif "opt" in args.net.lower():
                block_class_name = lm.model.model.decoder.layers[0].__class__.__name__
            device_map = infer_auto_device_map(lm.model, max_memory={i: args.max_memory for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
            load_checkpoint_in_model(lm.model, args.cal_save_path, device_map=device_map, dtype=torch.float32)
            lm.model.float()
            # spike_omniquant_opt(
            #     lm,
            #     args,
            #     cal_dataloader,
            #     act_scales,
            #     act_shifts,
            #     logger,
            # )
            # set_bit_state(lm.model, bit=4)
            spike_cal_8bit(
                lm,
                args,
                cal_dataloader,
                act_scales,
                act_shifts,
                logger,
            )
        elif args.load_cal:
            from accelerate import load_checkpoint_in_model, infer_auto_device_map
            if "llama" in args.net.lower():
                block_class_name = lm.model.model.layers[0].__class__.__name__
            elif "opt" in args.net.lower():
                block_class_name = lm.model.model.decoder.layers[0].__class__.__name__
            device_map = infer_auto_device_map(lm.model, max_memory={i: args.max_memory for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
            load_checkpoint_in_model(lm.model, args.cal_save_path, device_map=device_map, dtype=torch.float32)
        logger.info(time.time() - tick)
    if args.save_dir:
        # delete omni parameters
        for name, module in lm.model.named_modules():
            if isinstance(module, SpikeQuantLinear):
                del module.weight_quantizer.lowbound_factor
                del module.weight_quantizer.upbound_factor
            if isinstance(module,QuantLlamaDecoderLayer) or isinstance(module,QuantOPTDecoderLayer):
                if args.let:
                    del module.qkv_smooth_scale
                    del module.qkv_smooth_shift
                    del module.out_smooth_scale
                    del module.out_smooth_shift
                    del module.fc1_smooth_scale
                    del module.fc1_smooth_shift           
        lm.model.save_pretrained(args.save_dir)  
        lm.tokenizer.save_pretrained(args.save_dir) 
    for name, param in lm.model.named_parameters():
        if name.find('smooth') > -1 or name.find('bound_factor') > -1 or name.find('mask') > -1:
            param.requires_grad = False
        # print(f"{name}: {param.requires_grad}")
    #chents_add
    if 'opt' in args.model.lower():
        lm.tokenizer.bos_token_id = 0
    if "llama" in args.model.lower():
        lm.tokenizer.pad_token_id = 0
    #chents_train
    #train_args = parse_args()
    set_seed(args.seed)
    task = get_task(args.task_name)
    train_sets = task.sample_train_sets(num_train=args.num_train, num_dev=args.num_dev, num_eval=args.num_eval, num_train_sets=args.num_train_sets, seed=args.train_set_seed)
    #evaluate(lm, args,logger)
    def enable_quant(model, wq=True):
        set_quant_state(model, weight_quant=False, act_quant=True) 
        # for m in model.modules():
        #     if isinstance(m, (QuantLinear, QuantMatMul)):
        #         m.set_quant_state(weight_quant=False, act_quant=False)
    enable_quant(lm.model, wq=True)  
    framework = Framework(args, task, lm.model, lm.tokenizer)
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
                framework.model.cuda()
                if args.train:
                    framework.train(train_samples, dev_samples if dev_samples is not None else eval_samples)
                lm.model = framework.model  # Update the model after training
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
                    write_metrics_to_file(metrics, f"result/{args.model}/" + result_file_tag(args) + f"-trainset{train_set_id}.json" if args.result_file is None else args.result_file)

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
