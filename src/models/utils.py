import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import math
import random
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

sys.path.append("..")

from data.data_utils import language_identification
from .PersonaTransformer import TransformerwithPersona, Transformer

def seed_everything(seed):
  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)

def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)),k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0)
    if torch.cuda.is_available():
      np_mask = np_mask.cuda()
    return np_mask

def create_masks(src, trg):    
    src_mask = (src != 0).unsqueeze(-2)
    if trg is not None:
        trg_mask = (trg != 0).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = nopeak_mask(size)
        if trg.is_cuda:
            np_mask.cuda()
        trg_mask = trg_mask & np_mask
        
    else:
        trg_mask = None
    return src_mask, trg_mask

def get_model(args, src_vocab, trg_vocab, speaker_count, use_persona=True, generate_speaker_emb=False, bert_model=None, use_fame=True, use_alignment=True,use_linear_speaker_emb = False, no_speaker_id=False):
    
    assert args.d_model % args.heads == 0
    assert args.dropout < 1

    if use_persona:
      model = TransformerwithPersona(src_vocab, trg_vocab, speaker_count, args.d_model, args.n_layers, args.heads, \
                                     args.dropout, generate_speaker_emb=generate_speaker_emb, bert_model=bert_model,use_fame=use_fame, use_alignment=use_alignment, use_linear_speaker_emb=use_linear_speaker_emb, no_speaker_id=no_speaker_id)
    else:
      model = Transformer(src_vocab, trg_vocab, args.d_model, args.n_layers, args.heads, args.dropout, use_fame=use_fame, bert_model=bert_model)
       
    #try:
    #    print("loading pretrained weights...")
    #    model.load_state_dict(torch.load(f'{args.model_save_path}/model_weights.pth'))
    #except:
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p) 
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model

def calculate_metrics(eval_texts):
        
    tokenizer_ = AutoTokenizer.from_pretrained("sagorsarker/codeswitch-hineng-lid-lince")
    model_ = AutoModelForTokenClassification.from_pretrained("sagorsarker/codeswitch-hineng-lid-lince")
    #if torch.cuda.is_available():
    #    lid_model = pipeline('ner', model=model, tokenizer=tokenizer, device=0)
    #else:
    lid_model = pipeline('ner', model=model_, tokenizer=tokenizer_)
    
    hi_count = 0.0
    en_count = 0.0

    language_span = []

    total_count = 0.0
    
    repeat_index = 0.0

    for line in tqdm(eval_texts):
        lids = language_identification(lid_model, line)
        span_hi = 0
        span_en = 0
        prev = -1
        for token in list(lids.keys()):
            total_count +=1
            t = 0
            
            if lids[token] == 'hin':
                t = 1
            elif lids[token] == 'en':
                t = 2
                
            if t == 1:
                hi_count +=1
                if prev == 0:
                    span_hi += 1
                else:
                    language_span.append(span_en)
                    span_en = 0

                prev = 0
            elif t == 2:
                en_count +=1
                if prev == 1:
                    span_en += 1
                else:
                    language_span.append(span_hi)
                    span_hi = 0

                prev = 1
        
        values, counts = np.unique(np.array(line.split()), return_counts=True)
        #repeat_index += counts.max()/counts.sum()


    #repeat_index = repeat_index/len(eval_texts)
    try:
        p_hi = hi_count / total_count
    except:
        p_hi = 0
    
    try:
        p_en = en_count / total_count
    except:
        p_en = 0

    try:
        M_index = (1.0 - p_hi * p_hi - p_en * p_en) / (p_hi * p_hi + p_en * p_en)
    except:
        M_index = 0
    
    try:
        CMI = (total_count - max(hi_count, en_count))/total_count
    except:
        CMI = 0
    language_span = np.array(language_span)
    
    try:
        sd = np.std(language_span)
    except:
        sd = 0
    try:
        mean = np.mean(language_span)
    except:
        mean = 0
    #print (language_span, sd, mean)
    
    try:
        B = (sd - mean) / (sd + mean)
    except:
        B = 0

    unique, counts = np.unique(language_span, return_counts=True)

    hist = list(np.asarray((unique, counts)).T)
    total = len(language_span)
    try:
        LE = 0.0
        for (l, c) in hist:
            p_l = c*1.0 / total
            log_p = math.log(p_l)
            LE += p_l * log_p
        LE *= -1
    except:
        LE = 0
    
    return CMI, M_index, B, LE, repeat_index

def generate_transformer(
    model,
    bert_tokenizer,
    tokenizer,
    encode_text,
    decode_text,
    author_id,
    max_length=40,
    entry_length=30, #maximum number of words
    top_p=0.8,
    temperature=1.,
    eos_token='[SEP]', #<|endoftext|>,
    all_special_tokens = ['[CLS]','[PAD]','[SEP]', '[UNK]']
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    generated_num = 0
    generated_list = []
    
    speaker_ids = torch.LongTensor([author_id]).unsqueeze(0).to(device)

    filter_value = -float("Inf")

    with torch.no_grad():

        entry_finished = False
        src_generated = torch.tensor(bert_tokenizer.encode(encode_text, padding='max_length', max_length=max_length)[:max_length]).unsqueeze(0).to(device)
        trg_generated = torch.tensor(tokenizer.encode(decode_text)[:max_length][:-1]).unsqueeze(0).to(device)
        
        for i in range(entry_length):
            #print (src_generated, trg_generated)
            src_mask, trg_mask = create_masks(src_generated, trg_generated)
            logits, _, _, _ = model(src_generated, trg_generated, speaker_ids, src_mask.to(device), trg_mask.to(device))
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = filter_value

            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            trg_generated = torch.cat((trg_generated, next_token), dim=1)[:,:max_length]

            if next_token in tokenizer.encode(eos_token):
                entry_finished = True

            if entry_finished:

                generated_num = generated_num + 1

                output_list = list(trg_generated.squeeze().cpu().numpy())                
                output_text = tokenizer.decode(torch.tensor(output_list))
                generated_list.append(output_text)
                break

        if not entry_finished:
            output_list = list(trg_generated.squeeze().cpu().numpy())
            output_text = f"{tokenizer.decode(torch.tensor(output_list))}{eos_token}" 
            generated_list.append(output_text)
    
    generated_text = generated_list[0]
    
    for token in all_special_tokens:
        generated_text = generated_text.replace(token,'')
        
    return generated_text.strip() 

def generate_pretrained_transformer(
    model,
    tokenizer,
    encode_text,
    decode_text,
    entry_length=30, #maximum number of words
    top_p=0.8,
    temperature=1.,
    eos_token='[SEP]', #<|endoftext|>,
    all_special_tokens = ['[CLS]','[PAD]','[SEP]', '[UNK]']
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = model.to(device)
    model.eval()
    generated_num = 0
    generated_list = []

    filter_value = -float("Inf")

    with torch.no_grad():

        entry_finished = False

        #decode_text = encode_text + decode_text
        src_generated = torch.tensor(tokenizer.encode(decode_text)).unsqueeze(0).to(device)
        trg_generated = torch.tensor(tokenizer.encode(decode_text)).unsqueeze(0).to(device)
        
        for i in range(entry_length):
            #print (src_generated, trg_generated)
            trg_generated = trg_generated[:,:512]
            outputs = model(input_ids=trg_generated, labels=trg_generated)
            logits = outputs.logits
            #loss, logits = outputs[:2]
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = filter_value

            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            trg_generated = torch.cat((trg_generated, next_token), dim=1)[:,:model.config.max_length]

            if trg_generated.size()[1] >= model.config.max_length:
                entry_finished = True
                break

            if next_token in tokenizer.encode(eos_token):
                entry_finished = True

            if entry_finished:

                generated_num = generated_num + 1

                output_list = list(trg_generated.squeeze().cpu().numpy())                
                output_text = tokenizer.decode(torch.tensor(output_list))
                generated_list.append(output_text)
                break

        if not entry_finished:
            output_list = list(trg_generated.squeeze().cpu().numpy())
            output_text = f"{tokenizer.decode(torch.tensor(output_list))}{eos_token}" 
            generated_list.append(output_text)
    
    try:
        generated_text = generated_list[0]
    except:
        generated_text = decode_text
        
    for token in all_special_tokens:
        generated_text = generated_text.replace(str(token),'')
        
    return generated_text.strip() 