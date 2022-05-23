from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import pymongo

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from dataset import TextDataset, finetuneDataset, EvalDataset, lineDataset
from beam import Beam

from fuzzywuzzy import fuzz
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
from model import RNNModel

client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
db = client["test_input_activations"]
coordinates_col = db["coordinates"]
coordinates_col.remove()
train_col = db["activations"]
train_col.remove()


coordinates = []
#Save the coordinates of the selected neurons
activations = []
#Save the value of the neuron
layer_shapes = []
#Save the shape of each layer
layer_index = 0
#Save the current layer number


# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt='%m/%d/%Y %H:%M:%S',
#                     level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'rnn': (GPT2Config, RNNModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

def get_layers(model):
    layers = []

    def unfold_layer(model):
        layer_list = list(model.named_children())
        for item in layer_list:
            module = item[1]
            sublayer = list(module.named_children())
            sublayer_num = len(sublayer)

            if sublayer_num == 0:
                layers.append(module)
            elif isinstance(module, torch.nn.Module):
                unfold_layer(module)

    unfold_layer(model)

    return layers

def extract_shapes(self, input, output):
    global layer_shapes
    layer_shapes.append(output.shape)

def extract_neurons(self, input, output):
    global coordinates
    global activations
    global layer_index

    for index in coordinates:
        if index[0] == layer_index:
            position = index[1:]
            dim = len(output.shape)
            if dim == 2:
                activations.append(output[:,position[0]].unsqueeze(-1))
            elif dim == 3:
                activations.append(output[:,position[0],position[1]].unsqueeze(-1))
            elif dim == 4:
                activations.append(output[:,position[0],position[1],position[2]].unsqueeze(-1))
    layer_index += 1


def load_and_cache_examples(args, tokenizer, evaluate=False):
    if args.not_pretrain:
        dataset = finetuneDataset(tokenizer, args, logger, file_type='dev' if evaluate else 'train', 
                                block_size=args.block_size)
    else:
        dataset = TextDataset(tokenizer, args, logger, file_type='dev' if evaluate else 'train',
                                block_size=args.block_size)
    return dataset         

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def update_config(args, config):
    # config.n_positions = config.n_ctx = args.block_size
    config.vocab_size = args.vocab_size

def get_special_tokens(path):
    lits = json.load(open(path))
    tokens = ["<STR_LIT>", "<NUM_LIT>", "<CHAR_LIT>"]
    for lit in lits["str"]:
        tokens.append(f"<STR_LIT:{lit}>")
    for lit in lits["num"]:
        tokens.append(f"<NUM_LIT:{lit}>")
    for lit in lits["char"]:
        tokens.append(f"<CHAR_LIT:{lit}>")
    return tokens



def train(args, train_dataset, model, tokenizer, fh, pool):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        args.tensorboard_dir = os.path.join(args.output_dir, 'tensorboard')
        if not os.path.exists(args.tensorboard_dir):
            os.makedirs(args.tensorboard_dir)
    
    args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, drop_last=True)
    total_examples = len(train_dataset) * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    batch_size = args.batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    # if args.max_steps > 0:
    #     t_total = args.max_steps
    #     args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    if args.num_train_epochs > 0:
        t_total = total_examples // batch_size * args.num_train_epochs
    args.max_steps = t_total
    model.to(args.device)
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    # scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    # if os.path.exists(scheduler_last):
    #     scheduler.load_state_dict(torch.load(scheduler_last, map_location="cpu"))
    if os.path.exists(optimizer_last):
        logger.warning(f"Loading optimizer from {optimizer_last}")
        optimizer.load_state_dict(torch.load(optimizer_last, map_location="cpu"))   
    if args.local_rank == 0:
        torch.distributed.barrier()   
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
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank%args.gpu_per_node],
                                                          output_device=args.local_rank%args.gpu_per_node)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", total_examples )
    logger.info("  Num epoch = %d", t_total*batch_size//total_examples)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    
    global_step = args.start_step
    tr_loss, logging_loss,avg_loss,tr_nb = 0.0, 0.0, 0.0, global_step
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
 
    for idx in range(args.start_epoch, int(args.num_train_epochs)): 
        for step, batch in enumerate(train_dataloader):
            inputs, labels = (batch, batch)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            outputs = model(inputs, labels=labels)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
                
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if global_step % args.logging_steps == 0:
                    logger.info("  steps: %s  ppl: %s  lr: %s", global_step, round(avg_loss,5), scheduler.get_last_lr()[0])
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    logging_loss = tr_loss
                    tr_nb=global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, eval_when_training=True)
                        for key, value in results.items():
                            logger.info("  %s = %s", key, round(value,4))                    
                        output_dir = os.path.join(args.output_dir, '{}-{}-{}'.format(checkpoint_prefix, global_step, round(results['perplexity'],4)))
                    else:
                        output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    if args.model_type == "rnn":
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "model.pt"))
                    else:
                        model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    # _rotate_checkpoints(args, checkpoint_prefix)
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    if args.model_type == "rnn":
                        torch.save(model_to_save.state_dict(), os.path.join(last_output_dir, "model.pt"))
                    else:
                        model_to_save.save_pretrained(last_output_dir)
                    tokenizer.save_pretrained(last_output_dir)
                    idx_file = os.path.join(last_output_dir, 'idx_file.txt')
                    with open(idx_file, 'w', encoding='utf-8') as idxf:
                        idxf.write(str(0) + '\n')

                    torch.save(optimizer.state_dict(), os.path.join(last_output_dir, "optimizer.pt"))
                    # torch.save(scheduler.state_dict(), os.path.join(last_output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", last_output_dir)

                    step_file = os.path.join(last_output_dir, 'step_file.txt')
                    with open(step_file, 'w', encoding='utf-8') as stepf:
                        stepf.write(str(global_step) + '\n')
                    

            if args.max_steps > 0 and global_step > args.max_steps:
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            break

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    #logger.info("***** Running evaluation {} *****".format(prefix))
    #logger.info("  Num examples = %d", len(eval_dataset))
    #logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
       
    for batch in eval_dataloader:
        inputs, labels = (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": float(perplexity)
    }

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        #logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            #logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def eval_line_completion(args, model, tokenizer, file_type='test'):
    """
    Evaluate line level code completion on exact match and edit similarity.

    It is recommanded to use single GPU because it could not be batched.
    """
    
    layers = get_layers(model)
    #print(layers)
    def DecodeIds(idxs):
        codes = ""
        for idx in idxs:
            to_add = tokenizer.convert_ids_to_tokens(idx)
            if tokenizer.convert_ids_to_tokens(idx)[0] == '\u0120':
                if not codes.endswith(" "):
                    codes += " " + to_add[1:]
                else:
                    codes += to_add[1:]
            elif (
                idx in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id] or
                tokenizer.convert_ids_to_tokens(idx).startswith("<NUM_LIT")
            ):
                codes += " " + to_add + " "
            else:
                codes += to_add
        return codes.strip(" ")

    dataset = lineDataset(tokenizer, args, logger, file_type=file_type, block_size=args.block_size-100)
    test_sampler = SequentialSampler(dataset)
    test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=1)
    model.to(args.device)
    # model.zero_grad()
    model.eval()

    def repackage_hidden(h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)

    if args.langs == "python":
        break_ids = [tokenizer.sep_token_id]
    else:
        break_ids = [tokenizer.convert_tokens_to_ids('Ġ;'), tokenizer.convert_tokens_to_ids('Ġ}'), tokenizer.convert_tokens_to_ids('Ġ{')]
    preds = []
    gts = []
    edit_sim = 0.0
    em = 0.0
    for step, (inputs, gt) in enumerate(test_dataloader):
        inputs = inputs.to(args.device)
        with torch.no_grad():
            beam_size = 5
            m = torch.nn.LogSoftmax(dim=-1)
            outputs = model(inputs[:, :-1])[1]
            p = []       
            zero = torch.cuda.LongTensor(1).fill_(0)
            for i in range(inputs.shape[0]):
                if args.model_type == "rnn":
                    past_hidden = tuple(x[:, i:i+1].expand(-1, beam_size, -1).contiguous() for x in outputs)
                else:
                    past_hidden = [x[:, i:i+1].expand(-1, beam_size, -1, -1, -1) for x in outputs]
                beam = Beam(beam_size, inputs[i][-1].cpu().data, break_ids)
                input_ids = None
                for _ in range(100): 
                    if beam.done():
                        break
                    input_ids = beam.getCurrentState()
                    if args.model_type == "rnn":
                        outputs = model(input_ids, hidden=repackage_hidden(past_hidden))
                    else:
                        with torch.no_grad():
                            outputs = model(input_ids, past=past_hidden)
                    out = m(outputs[0][:, -1, :]).data
                    beam.advance(out)
                    if args.model_type == "rnn":
                        past_hidden = tuple(x.data.index_select(1, beam.getCurrentOrigin()).contiguous() for x in outputs[1])
                    else:
                        past_hidden = [x.data.index_select(1, beam.getCurrentOrigin()) for x in outputs[1]]
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:beam_size]

                pred = [torch.cat([x.view(-1) for x in p]+[zero]*(100-len(p))).view(1,-1) for p in pred]
                p.append(torch.cat(pred, 0).unsqueeze(0))
            p = torch.cat(p, 0)
            for pred in p:
                t = pred[0].cpu().numpy()
                t = t.tolist()
                if 0 in t:
                    t = t[:t.index(0)]
                if args.langs == "python":
                    text = DecodeIds(t).strip("<EOL>").strip()
                else:
                    text = DecodeIds(t).strip("{").strip()
                preds.append(text)
                gts.append(gt[0])
                edit_sim += fuzz.ratio(text, gt[0])
                em += 1 if text == gt[0] else 0
        if step % args.logging_steps == 0:
            logger.info(f"{step} are done!")

    saved_file = os.path.join(args.output_dir, "predictions_line.txt")
    with open(saved_file, "w") as f:
        for pred_text in preds:
            f.write(pred_text+"\n")
            
    logger.info(f"Test {len(preds)} samples")
    logger.info(f"Edit sim: {edit_sim/len(preds)}, EM: {em/len(preds)}")


def find_activations(args, model, tokenizer, file_type='test'):
    dataset = lineDataset(tokenizer, args, logger, file_type=file_type, block_size=args.block_size-100)
    test_sampler = SequentialSampler(dataset)
    test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=1)
    model.to(args.device)
    model.eval()
    
    global coordinates
    coordinates = []
    cur_index = []
    all_layer = [7, 16, 25, 34, 43, 52, 61, 70, 79, 88, 97, 106]
    for cur_layer_index in all_layer:
        tmp = 0
        for hid in range(768):
            cur_index = [cur_layer_index, tmp]
            cur_index.append(hid)
            coordinates.append(cur_index)
    mydict = {'coordinates': coordinates}
    coordinates_col.insert_one(mydict)

    for step, (inputs, rec_id, recitation_code, input_code_id, input_code) in enumerate(test_dataloader):
        print(step)
        layers = get_layers(model)
        #print(layers)
        inputs = inputs.to(args.device)
        #print(inputs)
        with torch.no_grad():
                outputs = model(inputs[:, :-1])[1]
        global activations
        activations = []
        global layer_index
        layer_index = 0
        for i in range(inputs.shape[0]):
            past_hidden = [x[:, i:i+1].expand(-1, 1, -1, -1, -1) for x in outputs]
            input_ids = inputs[:, -1]
            #print(input_ids)
            handles = []
            for l in layers:
                handle = l.register_forward_hook(extract_neurons)
                handles.append(handle)
            with torch.no_grad():
                outputs = model(input_ids, past = past_hidden)
            for handle in handles:
                handle.remove()
                
        act = torch.cat(activations, -1).detach().cpu().numpy()
        mydict = {"rec_id":rec_id[0], "recitation_code":recitation_code[0], "input_id":input_code_id.cpu().numpy().tolist()[0], "input":input_code[0], "activations": act[i,:].tolist()}
        train_col.insert_one(mydict)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data path.")
    parser.add_argument("--langs", default=None, type=str, required=True,
                        help="Languages to train, if all, train all languages in data_dir")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--model_type", default="gpt2", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--pretrain_dir", default="", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--config_dir", type=str,
                        help="config name. Required when training from scratch")
    parser.add_argument("--tokenizer_dir", type=str,
                        help="Pre-trained tokenizer dir. Required when training from scratch")
    parser.add_argument("--lit_file", type=str,
                        help="literals json file")
    parser.add_argument("--load_name", type=str, default="pretrained", 
                        help="Load pretrained model name")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=1024, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--eval_line", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=12, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=1000,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=5000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--not_pretrain', action='store_true',
                        help="use different dataset")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--node_index", type=int, default=-1,
                        help="node index if multi-node running")    
    parser.add_argument("--gpu_per_node", type=int, default=-1,
                        help="num of gpus per node")  
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    parser.add_argument('--log_file', type=str, default='')
    parser.add_argument('--tensorboard_dir', type=str)  
    
    pool = None
    args = parser.parse_args()

    # args.output_dir = os.path.join(args.output_dir, args.dataset)

    if args.model_type in ["bert", "roberta", "distilbert"] and not args.mlm:
        raise ValueError("BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
                         "flag (masked language modeling).")

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    logger.info("local_rank: %d, node_index: %d, gpu_per_node: %d"%(args.local_rank, args.node_index, args.gpu_per_node))
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.local_rank += args.node_index * args.gpu_per_node
        args.n_gpu = 1
    args.device = device
    # args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s, world size: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16,
                   torch.distributed.get_world_size() if args.local_rank != -1 else 1)


    fh = logging.FileHandler(args.log_file)
    logger.addHandler(fh)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if args.do_train and os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.pretrain_dir = os.path.join(checkpoint_last)
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} steps".format(checkpoint_last, args.start_step))

    # get special tokens
    special_tokens = get_special_tokens(args.lit_file)

    # Load pre-trained model
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    pretrained = args.pretrain_dir
    if pretrained:
        tokenizer = tokenizer_class.from_pretrained(pretrained, do_lower_case=args.do_lower_case, sep_token='<EOL>', bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>', additional_special_tokens=special_tokens)
        if args.model_type == "rnn":
            model = model_class(len(tokenizer), 768, 768, 1)
            model_last = os.path.join(pretrained, 'model.pt')
            if os.path.exists(model_last):
                logger.warning(f"Loading model from {model_last}")
                model.load_state_dict(torch.load(model_last, map_location="cpu")) 
        else:
            model = model_class.from_pretrained(pretrained)
            model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_dir, sep_token='<EOL>', bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>', additional_special_tokens=special_tokens)
        args.vocab_size = len(tokenizer)
        if args.model_type == "rnn":
            model = model_class(len(tokenizer), 768, 768, 1)
        else:
            config = config_class.from_pretrained(args.config_dir)
            model = model_class(config)
            model.resize_token_embeddings(len(tokenizer))


    model_parameters = model.parameters()
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info(f"Model has a total of {num_params} trainable parameters")

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    #if args.do_train:
    #    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

    #   global_step, tr_loss = train(args, train_dataset, model, tokenizer, fh, pool)
    #  logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    
    sampled_neurons = find_activations(args, model, tokenizer, file_type="test")
    
    '''
    # Only works on single GPU
    if args.eval_line:
        eval_line_completion(args, model, tokenizer, file_type="test")
    '''

if __name__ == "__main__":
    main()
