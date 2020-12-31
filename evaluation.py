"""
SOM-DST
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

from utils.data_utils import prepare_dataset, MultiWozDataset,map_state_to_ids
from utils.data_utils import make_slot_meta, domain2id, OP_SET, make_turn_label, postprocessing
from utils.eval_utils import compute_prf, compute_acc, per_domain_join_accuracy
from pytorch_transformers import BertTokenizer, BertConfig

from model import SomDST
from collections import Counter
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import random
import numpy as np
import os
import time
import argparse
import json
from copy import deepcopy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    ontology = json.load(open(os.path.join(args.data_root, args.ontology_data)))
    slot_meta, _ = make_slot_meta(ontology)
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=True)
    data = prepare_dataset(os.path.join(args.data_root, args.test_data),
                           tokenizer,
                           slot_meta, args.n_history, args.max_seq_length, args.op_code)

    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.dropout = 0.1
    op2id = OP_SET[args.op_code]
    model = SomDST(model_config, len(op2id), len(domain2id), op2id['update'])
    ckpt = torch.load(args.model_ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)

    model.eval()
    model.to(device)

    if args.eval_all:
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         False, False, False)
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         False, False, True)
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         False, True, False)
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         False, True, True)
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         True, False, False)
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         True, True, False)
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         True, False, True)
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         True, True, True)
    else:
        model_evaluation(model, data, tokenizer, slot_meta, 0, args.op_code,
                         args.gt_op, args.gt_p_state, args.gt_gen)

def op_evaluation_sketchy(op_prediction,op_ids):
    op_guess = 0.0
    op_correct = 0.0
    op_update_guess = 0.0
    op_update_correct = 0.0
    op_update_gold = 0.0
    for i, op_pred in enumerate(op_prediction):
        op_guess += 1
        if op_pred == op_ids[i]:
            op_correct += 1
            if op_ids[i] == 0:
                op_update_correct += 1
        if op_ids[i] == 0:
            op_update_gold += 1
        if op_prediction[i] == 0:
            op_update_guess += 1

    op_acc = op_correct / op_guess if op_guess != 0 else 0
    op_prec = op_update_correct / op_update_guess if op_update_guess != 0 else 0
    op_recall = op_update_correct / op_update_gold if op_update_gold != 0 else 0
    op_F1 = 2 * (op_prec * op_recall) / (op_prec + op_recall) if op_prec + op_recall != 0 else 0
    return op_acc,op_prec,op_recall,op_F1

def op_evaluation(start_prediction,end_prediction,gen_prediction,op_prediction,start_idx,end_idx,slot_ans_idx,op_ids,isopmask=False):
    gen_guess=0.0
    gen_correct=0.0
    op_guess=0.0
    op_correct=0.0
    op_update_guess=0.0
    op_update_correct=0.0
    op_update_gold=0.0


    for i,op_pred in enumerate(op_prediction):
        op_guess+=1
        if op_pred==op_ids[i]:
            op_correct+=1
            if op_ids[i]==0:
                op_update_correct+=1
        if op_ids[i]==0:
            gen_guess+=1
            op_update_gold+=1
            if start_idx[i]!=-1:
                gen_correct+=1*(start_idx[i]==start_prediction[i])*(end_idx[i]==end_prediction[i])
            else:
                gen_correct+=1*(gen_prediction[i]==slot_ans_idx[i])
        if op_prediction[i]==0:
            op_update_guess+=1

    gen_acc=gen_correct/gen_guess if gen_guess!=0 else 0
    op_acc=op_correct/op_guess if op_guess!=0 else 0
    op_prec=op_update_correct/op_update_guess if op_update_guess!=0 else 0
    op_recall=op_update_correct/op_update_gold if op_update_gold!=0 else 0
    op_F1=2*(op_prec*op_recall)/(op_prec+op_recall) if op_prec+op_recall!=0 else 0
    print(gen_correct)
    print(gen_guess)
    #print("Update score: operation precision: %.3f, operation_recall : %.3f,operation F1:%.3f"% (
        #op_prec, op_recall,op_F1))
    return gen_acc,op_acc,op_prec,op_recall,op_F1


def model_evaluation(model, test_data, tokenizer, slot_meta,epoch,slot_ans=None,  op_code='4',
                     is_gt_op=False, is_gt_p_state=False, is_gt_gen=False,eval_generate=True):


    model.eval()
    op2id = OP_SET[op_code]
    id2op = {v: k for k, v in op2id.items()}
    id2domain = {v: k for k, v in domain2id.items()}


    slot_turn_acc, joint_acc, slot_F1_pred, slot_F1_count = 0, 0, 0, 0
    final_joint_acc, final_count, final_slot_F1_pred, final_slot_F1_count = 0, 0, 0, 0
    op_acc, op_F1, op_F1_count = 0, {k: 0 for k in op2id}, {k: 0 for k in op2id}
    all_op_F1_count = {k: 0 for k in op2id}

    tp_dic = {k: 0 for k in op2id}
    fn_dic = {k: 0 for k in op2id}
    fp_dic = {k: 0 for k in op2id}


    # with open('samples.json','r') as f:
    #     err_ids=json.load(f)
    results = {}
    wall_times = []
    # err_ids={}
    last_slot_idx=[]
    slot_idx=[]
    gen_guess = 0.0
    gen_correct = 0.0
    op_guess = 0.0
    op_correct = 0.0
    op_update_guess = 0.0
    op_update_correct = 0.0
    op_update_gold = 0.0
    joint_correct=0.0
    for di, i in enumerate(test_data):
        # print()
        # if i.id not in err_ids.keys():
        #     continue
        if i.turn_id == 0:
            last_dialog_state = {}
            last_slot_idx=[-1]*len(slot_meta)

        if is_gt_p_state is False:
            i.last_dialog_state = deepcopy(last_dialog_state)
            i.make_instance(tokenizer, word_dropout=0.)
        else:  # ground-truth previous dialogue state
            last_dialog_state = deepcopy(i.gold_p_state)
            i.last_dialog_state = deepcopy(last_dialog_state)
            i.make_instance(tokenizer, word_dropout=0.)

        input_ids = torch.LongTensor([i.input_id]).to(device)
        input_mask = torch.FloatTensor([i.input_mask]).to(device)
        segment_ids = torch.LongTensor([i.segment_id]).to(device)
        state_position_ids = torch.LongTensor([i.slot_position]).to(device)
        # start_idx = torch.LongTensor([i.start_position]).to(device)
        # end_idx = torch.LongTensor([i.end_position]).to(device)
        slot_mask=torch.LongTensor([i.slot_mask]).to(device)
        slot_ans_idx=torch.LongTensor(i.gold_state_idx).to(device)
        # slot_ans_idx = torch.LongTensor(i.slot_ans_ids).to(device)
        # d_gold_op, generate_y, gold_state, generate_idx, slot_ans_idx= make_turn_label\
        #     (slot_meta, last_dialog_state, i.gold_state,
        #                                   tokenizer, slot_ans=slot_ans,op_code=op_code, dynamic=True)

        gold_op_ids = torch.LongTensor([i.op_ids]).to(device)
        pred_op_ids=torch.LongTensor([i.pred_op])

        start = time.perf_counter()
        MAX_LENGTH = 9
        d=None
        s=None
        g=None
        with torch.no_grad():
            # ground-truth state operation
            if eval_generate:
                start_logits, end_logits, gen_scores = model(input_ids=input_ids,
                                                                      token_type_ids=segment_ids,
                                                                      state_positions=state_position_ids,
                                                                      attention_mask=input_mask,
                                                                      slot_mask=slot_mask,
                                                                      max_value=0,
                                                                      op_ids=gold_op_ids,
                                                                      max_update=0)
            else:
                d, s = model(input_ids=input_ids,
                                token_type_ids=segment_ids,
                                state_positions=state_position_ids,
                                attention_mask=input_mask,
                                max_value=MAX_LENGTH,
                                op_ids=None)


        # if i.turn_id in err_ids[i.id]:
        #     print()

        # start_idx = start_idx.view(-1).cpu().detach().numpy().tolist()
        # end_idx = end_idx.view(-1).cpu().detach().numpy().tolist()
        start_idx=i.start_idx
        end_idx=i.end_idx
        slot_ans_idx=i.gold_state_idx
        start_prediction=start_logits.argmax(dim=-1).view(-1).cpu().detach().numpy().tolist()
        end_prediction = end_logits.argmax(dim=-1).view(-1).cpu().detach().numpy().tolist()
        #op_predictions = has_ans.argmax(dim=-1).view(-1).cpu().detach().numpy().tolist()
        op_predictions=pred_op_ids.view(-1).cpu().detach().numpy().tolist()
        gen_predictions = gen_scores.argmax(dim=-1).view(-1).cpu().detach().numpy().tolist()
        gold_op_ids = gold_op_ids.view(-1).cpu().detach().numpy().tolist()
        #slot_ans_idx = slot_ans_idx.view(-1).cpu().detach().numpy().tolist()
        op_guess += 1
        slot_idx=[-1]*len(last_slot_idx)
        iswrong=False
        for idx,op_pred in enumerate(op_predictions):
            if start_idx[idx]!=-1:
                if start_prediction[idx]!=0 and end_prediction[idx]!=0:
                    if op_pred==1:
                        slot_idx[idx]=last_slot_idx[idx]
                    else:
                        if start_prediction[idx]>end_prediction[idx] or (start_prediction[idx] !=start_idx[idx] or end_prediction[idx]!=end_idx[idx]):
                            slot_idx[idx] = -1
                        else:
                            slot_idx[idx]=slot_ans_idx[idx]
                elif start_prediction[idx]==0 and end_prediction[idx]==0:
                    slot_idx[idx]=last_slot_idx[idx]
                else:
                    if op_pred==1:
                        slot_idx[idx]=last_slot_idx[idx]
                    else:
                        slot_idx[idx]=-1
            else:
                if op_pred==1:
                    slot_idx[idx] = last_slot_idx[idx]
                else:
                    if gen_predictions[idx]==len(slot_ans[slot_meta[idx]])-2:
                        slot_idx[idx]=last_slot_idx[idx]
                    elif gen_predictions==len(slot_ans[slot_meta[idx]])-1:
                        slot_idx[idx]=-1
                    else:
                        slot_idx[idx]=gen_predictions[idx]

            if slot_idx[idx] != slot_ans_idx[idx]:
                iswrong=True
            if op_pred == gold_op_ids[idx]:
                op_correct += 1
                if gold_op_ids[idx] == 0:
                    op_update_correct += 1
            if gold_op_ids[idx] == 0:
                gen_guess += 1
                op_update_gold += 1
                if start_idx[idx] != -1:
                    gen_correct += 1 * (start_idx[idx] == start_prediction[idx]) * (end_idx[idx] == end_prediction[idx])
                else:
                    gen_correct += 1 * (gen_predictions[idx] == slot_ans_idx[idx])
            if op_pred == 0:
                op_update_guess += 1

        if not iswrong:
            joint_correct+=1
        last_slot_idx=slot_idx
    print(joint_correct/len(test_data))
    print(op_correct/(op_guess*30))
    op_recall=op_update_correct/op_update_gold
    op_prec=op_update_correct/op_update_guess
    print(op_prec)
    print(op_recall)
    print(2*(op_prec*op_recall)/(op_prec+op_recall))

    #     _, op_ids = s.view(-1, len(op2id)).max(-1)
    #
    #     if g.size(1) > 0:
    #         generated = g.squeeze(0).max(-1)[1].tolist()
    #     else:
    #         generated = []
    #
    #     if is_gt_op:
    #         pred_ops = [id2op[a] for a in gold_op_ids[0].tolist()]
    #     else:
    #         pred_ops = [id2op[a] for a in op_ids.tolist()]
    #     gold_ops = [id2op[a] for a in d_gold_op]
    #
    #     if is_gt_gen:
    #         # ground_truth generation
    #         gold_gen = {'-'.join(ii.split('-')[:2]): ii.split('-')[-1] for ii in i.gold_state}
    #     else:
    #         gold_gen = {}
    #
    #     if eval_generate:
    #         generated, last_dialog_state = postprocessing(slot_meta, pred_ops, last_dialog_state,
    #                                                       generated, tokenizer, op_code, gold_gen)
    #         pred_state = []
    #         for k, v in last_dialog_state.items():
    #             pred_state.append('-'.join([k, v]))
    #
    #         if set(pred_state) == set(i.gold_state):
    #             joint_acc += 1
    #
    #         key = str(i.id) + '_' + str(i.turn_id)
    #         results[key] = [pred_state, i.gold_state]
    #         # Compute prediction slot accuracy
    #         temp_acc = compute_acc(set(i.gold_state), set(pred_state), slot_meta)
    #         slot_turn_acc += temp_acc
    #         # Compute prediction F1 score
    #         temp_f1, temp_r, temp_p, count = compute_prf(i.gold_state, pred_state)
    #         slot_F1_pred += temp_f1
    #         slot_F1_count += count
    #         if i.is_last_turn:
    #             final_count += 1
    #             if set(pred_state) == set(i.gold_state):
    #                 final_joint_acc += 1
    #             final_slot_F1_pred += temp_f1
    #             final_slot_F1_count += count
    #
    #     end = time.perf_counter()
    #     wall_times.append(end - start)
    #
    #
    #
    #
    #
    #     # Compute operation accuracy
    #     match=0
    #     # for index in range(len(pred_ops)):
    #     #     if pred_ops[index]==gold_ops[index]:
    #
    #     temp_acc = sum([1 if p == g else 0 for p, g in zip(pred_ops, gold_ops)]) / len(pred_ops)
    #     op_acc += temp_acc
    #
    #
    #
    #     # Compute operation F1 score
    #     for p, g in zip(pred_ops, gold_ops):
    #         all_op_F1_count[g] += 1
    #         if p == g:
    #             tp_dic[g] += 1
    #             op_F1_count[g] += 1
    #         else:
    #             # if p in ("dontcare","delete") or g in ("dontcare","delete"):
    #                 # if i.id in err_ids.keys():
    #                 #     err_ids[i.id].append(i.turn_id)
    #                 # else:
    #                 #     err_ids[i.id]=[i.turn_id]
    #             fn_dic[g] += 1
    #             fp_dic[p] += 1
    # if eval_generate:
    #     joint_acc_score = joint_acc / len(test_data)
    #     turn_acc_score = slot_turn_acc / len(test_data)
    #     slot_F1_score = slot_F1_pred / slot_F1_count
    #
    #
    #
    #     final_joint_acc_score = final_joint_acc / final_count
    #     final_slot_F1_score = final_slot_F1_pred / final_slot_F1_count
    # latency = np.mean(wall_times) * 1000
    #
    # op_acc_score = op_acc / len(test_data)
    # op_F1_score = {}
    # for k in op2id.keys():
    #     tp = tp_dic[k]
    #     fn = fn_dic[k]
    #     fp = fp_dic[k]
    #     precision = tp / (tp+fp) if (tp+fp) != 0 else 0
    #     recall = tp / (tp+fn) if (tp+fn) != 0 else 0
    #     F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
    #     op_F1_score[k] = F1
    gen_acc = gen_correct / gen_guess if gen_guess != 0 else 0
    op_acc = op_correct / op_guess if op_guess != 0 else 0
    op_prec = op_update_correct / op_update_guess if op_update_guess != 0 else 0
    op_recall = op_update_correct / op_update_gold if op_update_gold != 0 else 0
    op_F1 = 2 * (op_prec * op_recall) / (op_prec + op_recall) if op_prec + op_recall != 0 else 0

    print("------------------------------")
    # print('op_code: %s, is_gt_op: %s, is_gt_p_state: %s, is_gt_gen: %s' % \
    #       (op_code, str(is_gt_op), str(is_gt_p_state), str(is_gt_gen)))
    #
    # print("Epoch %d op accuracy : " % epoch, op_acc_score)
    # print("Epoch %d op F1 : " % epoch, op_F1_score)
    # print("Epoch %d op hit count : " % epoch, op_F1_count)
    # print("Epoch %d op all count : " % epoch, all_op_F1_count)
    # scores = {'epoch': epoch,
    #           'op_acc': op_acc_score, 'op_f1': op_F1_score}
    if eval_generate:
        # print("Epoch %d joint accuracy : " % epoch, joint_acc)
        # print("Epoch %d slot turn accuracy : " % epoch, turn_acc_score)
        # print("Epoch %d slot turn F1: " % epoch, slot_F1_score)
        # print("Final Joint Accuracy : ", final_joint_acc_score)
        # print("Final slot turn F1 : ", final_slot_F1_score)
        scores = {'epoch': 0, 'joint_acc': joint_acc,
                  'slot_acc': 0.0, 'slot_f1': 0.0,
                  'op_acc': 0.0, 'op_f1': 0.0, 'final_slot_f1': 0.0}
        print(scores)
        #per_domain_join_accuracy(results, slot_meta)
    # print("Latency Per Prediction : %f ms" % latency)
    print("-----------------------------\n")
    json.dump(results, open('preds_%d.json' % epoch, 'w'))
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default='data/mwz2.1', type=str)
    parser.add_argument("--test_data", default='test_dials.json', type=str)
    parser.add_argument("--ontology_data", default='ontology.json', type=str)
    parser.add_argument("--vocab_path", default='assets/vocab.txt', type=str)
    parser.add_argument("--bert_config_path", default='assets/bert_config_base_uncased.json', type=str)
    parser.add_argument("--model_ckpt_path", default='outputs/model_best.bin', type=str)
    parser.add_argument("--n_history", default=1, type=int)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--op_code", default="4", type=str)

    parser.add_argument("--gt_op", default=False, action='store_true')
    parser.add_argument("--gt_p_state", default=False, action='store_true')
    parser.add_argument("--gt_gen", default=False, action='store_true')
    parser.add_argument("--eval_all", default=False, action='store_true')

    args = parser.parse_args()
    args.gt_p_state = False
    main(args)
