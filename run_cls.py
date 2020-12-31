# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import json

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
from models.SketchyReader import SketchyReader
import torch.nn as nn
from utils.data_utils import prepare_dataset, MultiWozDataset
# from utils.eval_utils import compute_prf, compute_acc, per_domain_join_accuracy
# from utils.ckpt_utils import download_ckpt, convert_ckpt_compatible
# from evaluation import model_evaluation
from utils.data_utils import make_slot_meta, domain2id, OP_SET, make_turn_label, postprocessing
from evaluation import model_evaluation,op_evaluation,op_evaluation_sketchy
from utils import helper
from transformers.configuration_albert import AlbertConfig
from transformers.tokenization_albert import AlbertTokenizer
# from transformers import (WEIGHTS_NAME, BertConfig,
#                                   BertForSequenceClassification, BertTokenizer,
#                                   RobertaConfig,
#                                   RobertaForSequenceClassification,
#                                   RobertaTokenizer,
#                                   XLMConfig, XLMForSequenceClassification,
#                                   XLMTokenizer, XLNetConfig,
#                                   XLNetForSequenceClassification,
#                                   XLNetTokenizer,
#                                   DistilBertConfig,
#                                   DistilBertForSequenceClassification,
#                                   DistilBertTokenizer,
#                                   AlbertConfig,
#                                   AlbertForSequenceClassification,
#                                   AlbertTokenizer,
#                                   XLMRobertaConfig,
#                                   XLMRobertaForSequenceClassification,
#                                   XLMRobertaTokenizer,
#                                 )

from transformer import AdamW, get_linear_schedule_with_warmup
from pytorch_transformers import BertTokenizer, AdamW, WarmupLinearSchedule, BertConfig

# from transformers import glue_compute_metrics as compute_metrics
# from transformers import glue_output_modes as output_modes
# from transformers import glue_processors as processors
# from transformers import glue_convert_examples_to_features as convert_examples_to_features
import sys
import csv
csv.field_size_limit(sys.maxsize)
logger = logging.getLogger(__name__)

# ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig,
#                                                                                 RobertaConfig, DistilBertConfig)), ())

# MODEL_CLASSES = {
#     'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
#     'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
#     'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
#     'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
#     'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
#     'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
#     'xlmroberta': (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
# }


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args,train_dataloader,dev_data_raw,slot_meta,model, tokenizer):
    """ Train the model """

    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()


    #train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    #train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    # som dataset process

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch[:-2])
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'labels':         batch[4]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = model_evaluation(model, dev_data_raw, tokenizer, slot_meta, step, args.op_code,
                     is_gt_op=False, is_gt_p_state=False, is_gt_gen=False)
                        for key, value in results.items():
                            eval_key = 'eval_{}'.format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs['learning_rate'] = learning_rate_scalar
                    logs['loss'] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{'step': global_step}}))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


# def evaluate(args, model, tokenizer, prefix=""):
#     # Loop to handle MNLI double evaluation (matched, mis-matched)
#     eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
#     eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)
#
#     results = {}
#     for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
#         eval_dataset,id_map = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
#
#         if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
#             os.makedirs(eval_output_dir)
#
#         args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
#         # Note that DistributedSampler samples randomly
#         eval_sampler = SequentialSampler(eval_dataset)
#         eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
#
#         # multi-gpu eval
#         if args.n_gpu > 1:
#             model = torch.nn.DataParallel(model)
#
#         # Eval!
#         logger.info("***** Running evaluation {} *****".format(prefix))
#         logger.info("  Num examples = %d", len(eval_dataset))
#         logger.info("  Batch size = %d", args.eval_batch_size)
#         eval_loss = 0.0
#         nb_eval_steps = 0
#         num_id = 0
#         preds = None
#         out_label_ids = None
#         key_map = {}
#         cnt_map = {}
#         for batch in tqdm(eval_dataloader, desc="Evaluating"):
#             model.eval()
#             batch = tuple(t.to(args.device) for t in batch)
#
#             with torch.no_grad():
#                 inputs = {'input_ids':      batch[0],
#                           'attention_mask': batch[1],
#                           'labels':         batch[3]}
#                 if args.model_type != 'distilbert':
#                     inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
#                 outputs = model(**inputs)
#                 tmp_eval_loss, logits = outputs[:2]
#
#                 eval_loss += tmp_eval_loss.mean().item()
#             nb_eval_steps += 1
#
#             logits = logits.detach().cpu().numpy()
#
#             for logit in logits:
#                 qas_id = id_map[num_id]
#                 if qas_id in key_map:
#                     logit_list = key_map[qas_id]
#                     logit_list[0] += logit[0]
#                     logit_list[1] += logit[1]
#                     cnt_map[qas_id] += 1
#                 else:
#                     cnt_map[qas_id] = 1
#                     key_map[qas_id] = [logit[0], logit[1]]
#                 num_id += 1
#
#             if preds is None:
#                 preds = logits
#                 out_label_ids = inputs['labels'].detach().cpu().numpy()
#             else:
#                 preds = np.append(preds, logits, axis=0)
#                 out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
#         print(len(preds))
#         eval_loss = eval_loss / nb_eval_steps
#         if args.output_mode == "classification":
#             preds = np.argmax(preds, axis=1)
#         elif args.output_mode == "regression":
#             preds = np.squeeze(preds)
#
#         #result = compute_metrics(eval_task, preds, out_label_ids)
#         results.update(result)
#
#         final_map = {}
#         for idx, key in enumerate(key_map):
#             key_list = key_map[key]
#             key_list[0] = key_list[0] / cnt_map[key]
#             key_list[1] = key_list[1] / cnt_map[key]
#             final_map[key] = key_list[1] - key_list[0]
#
#         with open(os.path.join(args.output_dir, "cls_score.json"), "w") as writer:
#             writer.write(json.dumps(final_map, indent=4) + "\n")
#
#         output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
#         with open(output_eval_file, "a") as writer:
#             logger.info("***** Eval results {} *****".format(prefix))
#             writer.write("***** Eval results %s *****\n" % (str(prefix)))
#             for key in sorted(result.keys()):
#                 logger.info("  %s = %s", key, str(result[key]))
#                 writer.write("%s = %s\n" % (key, str(result[key])))
#
#     return results


# def load_and_cache_examples(args, task, tokenizer, evaluate=False, predict=False):
#     if args.local_rank not in [-1, 0] and not evaluate:
#         torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
#
#     processor = processors[task]()
#     output_mode = output_modes[task]
#     # Load data features from cache or dataset file
#     # cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}_{}'.format(
#     #     'dev' if evaluate else 'train',
#     #     list(filter(None, args.model_name_or_path.split('/'))).pop(),
#     #     str(args.max_seq_length),str(args.doc_stride),
#     #     str(task)))
#     # if os.path.exists(cached_features_file) and not args.overwrite_cache:
#     #     logger.info("Loading features from cached file %s", cached_features_file)
#     #     features = torch.load(cached_features_file)
#     # else:
#     #     logger.info("Creating features from dataset file at %s", args.data_dir)
#     label_list = processor.get_labels()
#     if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta', 'xlmroberta']:
#         # HACK(label indices are swapped in RoBERTa pretrained model)
#         label_list[1], label_list[2] = label_list[2], label_list[1]
#     if predict:
#         examples = processor.get_test_examples(args.predict_file)
#     else:
#         examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
#     features, id_map = convert_examples_to_features(examples,
#                                             tokenizer,
#                                             label_list=label_list,
#                                             max_length=args.max_seq_length,
#                                             output_mode=output_mode,
#                                             pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
#                                             pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
#                                             pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,output_feature=True,
#     )
#     # if args.local_rank in [-1, 0]:
#     #     logger.info("Saving features into cached file %s", cached_features_file)
#     #     torch.save(features, cached_features_file)
#
#     if args.local_rank == 0 and not evaluate:
#         torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
#
#     # Convert to Tensors and build dataset
#     all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
#     all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
#     all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
#     if output_mode == "classification":
#         all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
#     elif output_mode == "regression":
#         all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
#
#     dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
#     return dataset,id_map

def masked_cross_entropy_for_value(logits, target,mask=None,pad_idx=-1):
    if mask is None:
        sample_mask = target.ne(pad_idx)
        mask=logits.eq(0)
    target=target.masked_fill(sample_mask==0,0)
    logits=logits.masked_fill(mask,1)
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = torch.log(logits_flat)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    # if mask is not None:
    sample_num=sample_mask.sum().float()
    losses = losses * sample_mask.float()
    loss = (losses.sum() / sample_num) if sample_num !=0 else 0
    return loss

# [SLOT], [NULL], [EOS]
def addSpecialTokens(tokenizer,specialtokens):
    special_key = "additional_special_tokens"
    tokenizer.add_special_tokens({special_key: specialtokens})

def fixontology(ontology):
    ontology['hotel-type'].append('none')
    ontology['restaurant-area'].append('none')
    ontology['attraction-area'].append('none')
    for k in ontology.keys():
        if 'day' in k:
            ontology[k].append('none')
        if "do n't care" not in ontology[k]:
            ontology[k].append("do n't care")
        ontology[k].append('[noans]')
        ontology[k].append('[negans]')
    return ontology

def mask_ans_vocab(ontology,slot_meta,tokenizer):
    ans_vocab = []
    max_anses = 0
    max_anses_length = 0
    change_k=[]
    for k in ontology.keys():
        if (' range' in k) or (' at' in  k) or (' by' in k):
            change_k.append(k)
        # fix_label(ontology[k])
    for key in change_k:
        new_k=key.replace(' ','')
        ontology[new_k]=ontology[key]
        del ontology[key]
    for s in slot_meta:
        v_list = ontology[s]
        slot_anses = []
        for v in v_list:
            ans = tokenizer.encode(v)
            max_anses_length = max(max_anses_length, len(ans))
            slot_anses.append(ans)
        max_anses = max(max_anses, len(slot_anses))
        ans_vocab.append(slot_anses)
    for s in ans_vocab:
        for ans in s:
            gap = max_anses_length - len(ans)
            ans += [0] * gap
        gap = max_anses - len(s)
        s += [[0] * max_anses_length] * gap
    ans_vocab = torch.LongTensor(ans_vocab)
    return ans_vocab

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_type", default='albert', type=str,
                        help="Model type selected in the list: ")
    parser.add_argument("--model_name_or_path", default='pretrained_models/albert_large/', type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--task_name", default='squad', type=str,
                        help="The name of the task to train selected in the list: ")
    parser.add_argument("--output_dir", default="saved_models/", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    # parser.add_argument("--max_seq_length", default=128, type=int,
    #                     help="The maximum total input sequence length after tokenization. Sequences longer "
    #                          "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", default=True,action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", default=True,action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",default=True, action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")     
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', default=True,action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--local_rank", type=int, default=-1,
                         help="For distributed training: local_rank")
    #parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    # parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    #DSTPARA
    parser.add_argument("--data_root", default='data/mwz2.1/', type=str)
    parser.add_argument("--train_data", default='train_dials.json', type=str)
    parser.add_argument("--dev_data", default='test_dials.json', type=str)
    parser.add_argument("--test_data", default='train_dials.json', type=str)
    parser.add_argument("--ontology_data", default='ontology.json', type=str)
    parser.add_argument("--vocab_path", default='assets/vocab.txt', type=str)
    # parser.add_argument("--bert_config_path", default='assets/bert_config_base_uncased.json', type=str)
    # parser.add_argument("--bert_ckpt_path", default='assets/bert-base-uncased-pytorch_model.bin', type=str)
    parser.add_argument("--save_dir", default='saved_models', type=str)

    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--enc_warmup", default=0.01, type=float)
    parser.add_argument("--dec_warmup", default=0.01, type=float)
    parser.add_argument("--enc_lr", default=1e-5, type=float)
    parser.add_argument("--base_lr", default=1e-5, type=float)
    parser.add_argument("--dec_lr", default=1e-5, type=float)
    parser.add_argument("--n_epochs", default=10, type=int)
    parser.add_argument("--eval_epoch", default=1, type=int)

    parser.add_argument("--op_code", default="2", type=str)
    parser.add_argument("--slot_token", default="[SLOT]", type=str)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float)
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float)
    parser.add_argument("--decoder_teacher_forcing", default=0.5, type=float)
    parser.add_argument("--word_dropout", default=0.1, type=float)
    parser.add_argument("--not_shuffle_state", default=True, action='store_true')
    parser.add_argument("--shuffle_p", default=0.5, type=float)
    parser.add_argument("--n_history", default=1, type=int)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--msg", default=None, type=str)
    parser.add_argument("--exclude_domain", default=False, action='store_true')




    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    args.shuffle_state = False if args.not_shuffle_state else True
    # Setup distant debugging if needed
    # if args.server_ip and args.server_port:
    #     # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
    #     import ptvsd
    #     print("Waiting for debugger attach")
    #     ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
    #     ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)
    def worker_init_fn(worker_id):
        np.random.seed(args.random_seed + worker_id)
    # Prepare GLUE task

    n_gpu = 0
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()

    args.task_name = args.task_name.lower()
    # if args.task_name not in processors:
    #     raise ValueError("Task not found: %s" % (args.task_name))
    # processor = processors[args.task_name]()
    # args.output_mode = output_modes[args.task_name]
    # label_list = processor.get_labels()
    # num_labels = len(label_list)
    num_labels=2
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    ontology = json.load(open(args.data_root+args.ontology_data))

    slot_meta, slot_ans_wh = make_slot_meta(ontology)


    op2id = OP_SET[args.op_code]
    rng = random.Random(args.random_seed)
    print(op2id)
    tokenizer = AlbertTokenizer.from_pretrained(args.model_name_or_path + "spiece.model")
    # model = AlbertModel.from_pretrained(args.model_name_or_path+"pytorch_model.bin",config=args.model_name_or_path+"config.json")
    addSpecialTokens(tokenizer,['[SLOT]','[NULL]','[EOS]'])
    args.vocab_size=len(tokenizer)
    ontology = fixontology(ontology)
    ans_vocab=mask_ans_vocab(ontology, slot_meta, tokenizer)
    # train_data_raw,_,_ = prepare_dataset(data_path=args.data_root+args.train_data,
    #                                  tokenizer=tokenizer,
    #                                  slot_meta=slot_meta,
    #                                  n_history=args.n_history,
    #                                  max_seq_length=args.max_seq_length,
    #                                  op_code=args.op_code,
    #                                 slot_ans=ontology)
    #
    # train_data = MultiWozDataset(train_data_raw,
    #                              tokenizer,
    #                              slot_meta,
    #                              args.max_seq_length,
    #                              rng,
    #                              ontology,
    #                              args.word_dropout,
    #                              args.shuffle_state,
    #                              args.shuffle_p)
    # print("# train examples %d" % len(train_data_raw))
    #
    #
    #
    # dev_data_raw,idmap,_= prepare_dataset(data_path=args.data_root+args.dev_data,
    #                                tokenizer=tokenizer,
    #                                slot_meta=slot_meta,
    #                                n_history=args.n_history,
    #                                max_seq_length=args.max_seq_length,
    #                                op_code=args.op_code,
    #                                 slot_ans=ontology)
    # print("# dev examples %d" % len(dev_data_raw))
    #
    # dev_data = MultiWozDataset(dev_data_raw,
    #                              tokenizer,
    #                              slot_meta,
    #                              args.max_seq_length,
    #                              rng,
    #                              ontology,
    #                              args.word_dropout,
    #                              args.shuffle_state,
    #                              args.shuffle_p)
    #
    # test_data_raw,_,_= prepare_dataset(data_path=args.data_root+args.test_data,
    #                                 tokenizer=tokenizer,
    #                                 slot_meta=slot_meta,
    #                                 n_history=args.n_history,
    #                                 max_seq_length=args.max_seq_length,
    #                                 op_code=args.op_code,
    #                                    slot_ans=ontology)
    # print("# test examples %d" % len(test_data_raw))
    # test_data = MultiWozDataset(test_data_raw,
    #                              tokenizer,
    #                              slot_meta,
    #                              args.max_seq_length,
    #                              rng,
    #                              ontology,
    #                              args.word_dropout,
    #                              args.shuffle_state,
    #                              args.shuffle_p)
    # test_sampler = RandomSampler(test_data)
    # test_dataloader = DataLoader(test_data,
    #                               sampler=test_sampler,
    #                               batch_size=args.batch_size,
    #                               collate_fn=test_data.collate_fn,
    #                               num_workers=args.num_workers,
    #                               worker_init_fn=worker_init_fn)
    #
    # train_sampler = RandomSampler(train_data)
    # train_dataloader = DataLoader(train_data,
    #                               sampler=train_sampler,
    #                               batch_size=args.batch_size,
    #                               collate_fn=train_data.collate_fn,
    #                               num_workers=args.num_workers,
    #                               worker_init_fn=worker_init_fn)
    #
    # dev_sampler = RandomSampler(dev_data)
    # dev_dataloader = DataLoader(dev_data,
    #                               sampler=dev_sampler,
    #                               batch_size=args.batch_size,
    #                               collate_fn=dev_data.collate_fn,
    #                               num_workers=args.num_workers,
    #                               worker_init_fn=worker_init_fn)

    # args.model_type = args.model_type.lower()
    #config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
    #                                       num_labels=num_labels,
    #                                       finetuning_task=args.task_name,
    #                                       cache_dir=args.cache_dir if args.cache_dir else None)
    # tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    #                                             do_lower_case=args.do_lower_case,
    #                                             cache_dir=args.cache_dir if args.cache_dir else None)
    # model = model_class.from_pretrained(args.model_name_or_path,
    #                                     from_tf=bool('.ckpt' in args.model_name_or_path),
    #                                     config=config,
    #                                     cache_dir=args.cache_dir if args.cache_dir else None)
    #
    # if args.local_rank == 0:
    #     torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # model_config = AlbertConfig.from_json_file(args.bert_config_path)
    # model_config.dropout = args.dropout
    # model_config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
    # model_config.hidden_dropout_prob = args.hidden_dropout_prob

    model = SketchyReader(args, len(op2id), len(domain2id), op2id['update'])
    ckpt = torch.load("saved_models/model_best_cls.bin", map_location='cpu')
    model.load_state_dict(ckpt)
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)


    # Training
    # if args.do_train:
    #     #train_dataset,id_map = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
    #     global_step, tr_loss = train(args,train_dataloader,dev_data_raw,slot_meta,model, tokenizer)
    #     logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    #

    # # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    # if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    #     # Create output directory if needed
    #     if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
    #         os.makedirs(args.output_dir)
    #
    #     logger.info("Saving model checkpoint to %s", args.output_dir)
    #     # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    #     # They can then be reloaded using `from_pretrained()`
    #     model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    #     model_to_save.save_pretrained(args.output_dir)
    #     tokenizer.save_pretrained(args.output_dir)
    #
    #     # Good practice: save your training arguments together with the trained model
    #     torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
    #
    #     # Load a trained model and vocabulary that you have fine-tuned
    #     model = model_class.from_pretrained(args.output_dir)
    #     tokenizer = tokenizer_class.from_pretrained(args.output_dir)
    #     model.to(args.device)


    # Evaluation
    # results = {}
    #if args.do_eval and args.local_rank in [-1, 0]:
    #     tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    #     checkpoints = [args.output_dir]
    #     if args.eval_all_checkpoints:
    #         checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
    #         logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    #     logger.info("Evaluate the following checkpoints: %s", checkpoints)
    #     for checkpoint in checkpoints:
    #         global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
    #         prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
    #
    #         model = model_class.from_pretrained(checkpoint)
    #         model.to(args.device)
    #         result = (args, model, tokenizer, prefix=prefix)
    #         result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
    #         results.update(result)
    #
    # if args.do_predict and args.local_rank in [-1, 0]:
    #     checkpoint = args.model_name_or_path
    #     model = model_class.from_pretrained(checkpoint, force_download=True)
    #     model.to(args.device)
    #
    #     eval_task = args.task_name
    #     eval_dataset, id_map = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True, predict=True)
    #
    #     args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    #     # Note that DistributedSampler samples randomly


    # if not os.path.exists(args.bert_ckpt_path):
    #     args.bert_ckpt_path = download_ckpt(args.bert_ckpt_path, args.bert_config_path, 'assets')
    #
    # ckpt = torch.load(args.bert_ckpt_path, map_location='cpu')
    # model.encoder.bert.load_state_dict(ckpt)

    # re-initialize added special tokens ([SLOT], [NULL], [EOS])
    # model.encoder.bert.embeddings.word_embeddings.weight.data[1].normal_(mean=0.0, std=0.02)
    # model.encoder.bert.embeddings.word_embeddings.weight.data[2].normal_(mean=0.0, std=0.02)
    # model.encoder.bert.embeddings.word_embeddings.weight.data[3].normal_(mean=0.0, std=0.02)
    # model.to(device)
    num_train_steps=0
    #num_train_steps = int(len(train_data_raw) / args.batch_size * args.n_epochs)
    bert_params_ids = list(map(id, model.albert.parameters()))
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    enc_param_optimizer = list(model.named_parameters())
    enc_optimizer_grouped_parameters = [
        {'params': [p for n, p in enc_param_optimizer if (id(p) in bert_params_ids and not any(nd in n for nd in no_decay))], 'weight_decay': 0.0,'lr':args.enc_lr},
        {'params': [p for n, p in enc_param_optimizer if id(p) in bert_params_ids and any(nd in n for nd in no_decay)], 'weight_decay': 0.0,'lr':args.enc_lr},
        {'params': [p for n, p in enc_param_optimizer if id(p) not in bert_params_ids and not any(nd in n for nd in no_decay)],'weight_decay': 0.0,'lr':args.base_lr},
        {'params': [p for n, p in enc_param_optimizer if id(p) not in bert_params_ids and any(nd in n for nd in no_decay)], 'weight_decay':0.0,'lr':args.base_lr}]

    enc_optimizer = AdamW(enc_optimizer_grouped_parameters, lr=args.base_lr)
    enc_scheduler = WarmupLinearSchedule(enc_optimizer, int(num_train_steps * args.enc_warmup),
                                         t_total=num_train_steps)

    # dec_param_optimizer = list(model.decoder.parameters())
    # dec_optimizer = AdamW(dec_param_optimizer, lr=args.dec_lr)
    # dec_scheduler = WarmupLinearSchedule(dec_optimizer, int(num_train_steps * args.dec_warmup),
    #                                      t_total=num_train_steps)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # train_sampler = RandomSampler(train_data)
    # train_dataloader = DataLoader(train_data,
    #                               sampler=train_sampler,
    #                               batch_size=args.batch_size,
    #                               collate_fn=train_data.collate_fn,
    #                               num_workers=args.num_workers,
    #                               worker_init_fn=worker_init_fn)

    loss_fnc = nn.CrossEntropyLoss()
    best_score = {'epoch': 0,'op_acc':0,'op_f1': 0}
    # model.eval()
    file_logger = helper.FileLogger(args.save_dir + '/log_cls.txt',
                                    header="# epoch\ttrain_loss\tdev_oscore\tdev_opp\tdev_opr\tdev_f1\tbest_oscore\tbest_opf1")

    # for epoch in range(args.n_epochs):
    #     batch_loss = []
    #     # model.eval()
    #     model.train()
    #     for step, batch in enumerate(train_dataloader):
    #         batch = [b.to(device) if not isinstance(b, int) and not isinstance(b,list) else b for b in batch]
    #         input_ids, input_mask, slot_mask, segment_ids, state_position_ids, op_ids, domain_ids, gen_ids, start_position, end_position, max_value, max_update, slot_ans_ids, start_idx, end_idx ,sid = batch
    #         batch_size = input_ids.shape[0]
    #         seq_lens = input_ids.shape[1]
    #         if rng.random() < args.decoder_teacher_forcing:  # teacher forcing
    #             teacher = gen_ids
    #         else:
    #             teacher = None
    #
    #         domain_score,has_ans = model(input_ids=input_ids,
    #                                                               token_type_ids=segment_ids,
    #                                                               state_positions=state_position_ids,
    #                                                               attention_mask=input_mask,
    #                                                               op_ids=op_ids,
    #                                                               max_update=max_update)
    #
    #         loss_ans = loss_fnc(has_ans.view(-1, len(op2id)), op_ids.view(-1))
    #         # start_loss = loss_fnc(start_logits.view(-1,seq_lens),start_position.view(-1,seq_lens))
    #         # end_loss = loss_fnc(end_logits.view(-1, seq_lens), end_position.view(-1,seq_lens))
    #
    #         # loss_g = masked_cross_entropy_for_value(gen_scores.contiguous(),
    #         #                                         slot_ans_ids.contiguous(),
    #         #                                         )
    #         # loss_s = masked_cross_entropy_for_value(start_logits.contiguous(),
    #         #                                         start_idx.contiguous(),
    #         #                                         )
    #         # loss_e = masked_cross_entropy_for_value(end_logits.contiguous(),
    #         #                                         end_idx.contiguous())
    #
    #         loss = loss_ans
    #         if args.exclude_domain is not True:
    #             loss_d = loss_fnc(domain_score.view(-1, len(domain2id)), domain_ids.view(-1))
    #             loss = loss + loss_d
    #         # loss=loss+start_loss+end_loss
    #         batch_loss.append(loss.item())
    #         # with torch.autograd.detect_anomaly():
    #         #     loss.backward()
    #         loss.backward()
    #         enc_optimizer.step()
    #         enc_scheduler.step()
    #         # dec_optimizer.step()
    #         # dec_scheduler.step()
    #         model.zero_grad()
    #
    #         if step % 100 == 0:
    #             if args.exclude_domain is not True:
    #                 print("[%d/%d] [%d/%d] mean_loss : %.3f" \
    #                       % (epoch + 1, args.n_epochs, step,
    #                          len(train_dataloader), np.mean(batch_loss)))
    #             else:
    #                 print("[%d/%d] [%d/%d] mean_loss : %.3f, state_loss : %.3f," \
    #                       % (epoch + 1, args.n_epochs, step,
    #                          len(train_dataloader), np.mean(batch_loss),
    #                          loss.item()))
    #             batch_loss = []
    #
    #         if (step + 1) % 1000 == 0:
    #             model.eval()
    #             has_ans_predictions = []
    #             has_ans_labels = []
    #             for step, batch in enumerate(dev_dataloader):
    #                 batch = [b.to(device) if not isinstance(b, int) and not isinstance(b,list) else b for b in batch]
    #                 input_ids, input_mask, slot_mask, segment_ids, state_position_ids, op_ids, domain_ids, gen_ids, start_position, end_position, max_value, max_update, slot_ans_ids, start_idx, end_idx,sid = batch
    #                 batch_size = input_ids.shape[0]
    #                 seq_lens = input_ids.shape[1]
    #                 if rng.random() < args.decoder_teacher_forcing:  # teacher forcing
    #                     teacher = gen_ids
    #                 else:
    #                     teacher = None
    #
    #                 domain_score, has_ans = model(input_ids=input_ids,
    #                                               token_type_ids=segment_ids,
    #                                               state_positions=state_position_ids,
    #                                               attention_mask=input_mask,
    #                                               op_ids=op_ids,
    #                                               max_update=max_update)
    #                 has_ans_predictions += has_ans.argmax(dim=-1).view(-1).cpu().detach().numpy().tolist()
    #                 has_ans_labels += op_ids.view(-1).cpu().detach().numpy().tolist()
    #
    #             op_acc, op_prec, op_recall, op_F1 = op_evaluation_sketchy(has_ans_predictions, has_ans_labels)
    #             # eval_res = model_evaluation(model, dev_data_raw, tokenizer, slot_meta, epoch + 1, args.op_code)
    #             file_logger.log(
    #                 "{}\t{:.6f}\t{:.6f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(epoch, loss, op_acc, op_prec,
    #                                                                                     op_recall, op_F1,
    #                                                                                     max(op_acc, best_score['op_acc']),
    #                                                                                     max(op_F1, best_score['op_f1'])))
    #             if op_F1 > best_score['op_f1']:
    #                 best_score['op_acc'] = op_acc
    #                 best_score['op_f1'] = op_F1
    #                 model_to_save = model.module if hasattr(model, 'module') else model
    #                 save_path = os.path.join(args.save_dir, 'model_best_cls.bin')
    #                 torch.save(model_to_save.state_dict(), save_path)
    #                 file_logger.log("new best model saved at epoch {}: {:.2f}\t{:.2f}" \
    #                                 .format(epoch, op_acc * 100, op_F1 * 100))
    #             print(
    #                 "Best Score : operation_accurate : %.3f,operation_precision : %.3f, operation_recall:%.3f,operation_F1 : %.3f" % (
    #                 op_acc, op_prec, op_recall, best_score['op_f1']))
    #             print("\n")

    # print("Test using best model...")
    # best_epoch = best_score['epoch']
    #ckpt_path = os.path.join(args.save_dir, 'model_best.bin')
    #model = Verifier(args, len(op2id), len(domain2id), op2id['update'], args.exclude_domain)

    test_data_raw, _, _ = prepare_dataset(data_path=args.data_root + args.test_data,
                                          tokenizer=tokenizer,
                                          slot_meta=slot_meta,
                                          n_history=args.n_history,
                                          max_seq_length=args.max_seq_length,
                                          op_code=args.op_code,
                                          slot_ans=ontology)
    print("# test examples %d" % len(test_data_raw))
    test_data = MultiWozDataset(test_data_raw,
                                tokenizer,
                                slot_meta,
                                args.max_seq_length,
                                rng,
                                ontology,
                                args.word_dropout,
                                args.shuffle_state,
                                args.shuffle_p)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data,
                                 sampler=test_sampler,
                                 batch_size=args.batch_size,
                                 collate_fn=test_data.collate_fn,
                                 num_workers=args.num_workers,
                                 worker_init_fn=worker_init_fn)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(test_data_raw))
    logger.info("  Batch size = %d", args.batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    num_id = 0
    key_map = {}
    cnt_map = {}

    score_ext_map = {}
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        model.eval()
        batch = [b.to(device) if not isinstance(b, int) and not isinstance(b,list) else b for b in batch]
        input_ids, input_mask, slot_mask, segment_ids, state_position_ids, op_ids, domain_ids, gen_ids, start_position, end_position, max_value, max_update, slot_ans_ids, start_idx, end_idx,sid = batch
        batch_size = input_ids.shape[0]
        seq_lens = input_ids.shape[1]
        if rng.random() < args.decoder_teacher_forcing:  # teacher forcing
            teacher = gen_ids
        else:
            teacher = None

        domain_score, has_ans = model(input_ids=input_ids,
                                      token_type_ids=segment_ids,
                                      state_positions=state_position_ids,
                                      attention_mask=input_mask,
                                      op_ids=op_ids,
                                      max_update=max_update)

        score_ext=has_ans.cpu().detach().numpy().tolist()
        for i,sd in enumerate(score_ext):
            score_ext_map[sid[i]]=sd
    with open(os.path.join(args.output_dir, "cls_score_train.json"), "w") as writer:
        writer.write(json.dumps(score_ext_map, indent=4) + "\n")
    # for logit in logits:
    #             qas_id = idmap[num_id]
    #             if qas_id in key_map:
    #                 logit_list = key_map[qas_id]
    #                 logit_list[0] += logit[0]
    #                 logit_list[1] += logit[1]
    #                 cnt_map[qas_id] += 1
    #             else:
    #                 cnt_map[qas_id] = 1
    #                 key_map[qas_id] = [logit[0], logit[1]]
    #             num_id += 1

    # final_map = {}
    # for idx, key in enumerate(key_map):
    #     key_list = key_map[key]
    #     key_list[0] = key_list[0] / cnt_map[key]
    #     key_list[1] = key_list[1] / cnt_map[key]
    #     # key_list[0] = key_list[0]
    #     # key_list[1] = key_list[1]
    #     # final_map[key] = key_list[1]
    #     # final_map[key] = key_list[1]*2
    #     final_map[key] = key_list[1] - key_list[0]
    #
    # with open(os.path.join(args.output_dir, "cls_score.json"), "w") as writer:
    #     writer.write(json.dumps(final_map, indent=4) + "\n")

if __name__ == "__main__":
    main()
