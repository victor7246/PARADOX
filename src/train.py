import os
import sys
import math
import random
import re
import argparse
import copy
from copy import deepcopy as cp
from collections import OrderedDict
import dotenv
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.init as init
from torch.autograd.function import InplaceFunction
from torch.autograd import Variable

from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, BertTokenizer, AutoModelForSequenceClassification, AutoModel

import wandb

from tqdm import tqdm

from data.data_utils import remove_emojis, remove_html, remove_email, clean_tweets, calculate_CMI, language_identification
from data.custom_tokenizers import custom_wp_tokenizer
from data.datasets import TransformerDataset
from models.utils import calculate_metrics, get_model, create_masks, generate_transformer
from trainer import meta_trainer, trainer_without_metalearning

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

if __name__ == '__main__':
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(prog='Trainer',conflict_handler='resolve')

    parser.add_argument('--data_file', type=str, default='../data/twitter_data.csv', required=False,
                        help='train data')

    parser.add_argument('--use_fame', action='store_true',
                        help='Use FAME attention')
    parser.add_argument('--use_persona', action='store_true',
                        help='Use persona model')
    parser.add_argument('--use_rezero', action='store_true',
                        help='Use rezero')
    parser.add_argument('--use_alignment', action='store_true',
                        help='Use alignment module in')
    parser.add_argument('--use_linear_speaker_emb', action='store_true',
                        help='Use linear persona encoder module in')
    parser.add_argument('--teacher_model', type=str, default='google/muril-base-cased', required=False,
                        help='teacher model name (Huggingface pretrained models')
    #parser.add_argument('--use_teacher_encoder', action='store_true',
    #                    help='use teacher model in encoding')
    parser.add_argument('--use_distillation', action='store_true',
                        help='Use distillation: True to enable, False for vanilla training')
    parser.add_argument('--use_meta_distillation', action='store_true',
                        help='Use meta distillation: True to enable, False for vanilla training')
    parser.add_argument('--generate_random_speaker', action='store_true',
                        help='Generate random speaker persona')
    parser.add_argument('--no_speaker_id', action='store_true',
                        help='No speaker token')

    parser.add_argument('--max_text_len', type=int, default=60, required=False,
                        help='maximum length of text')
    parser.add_argument('--n_layers', type=int, default=6, required=False,
                        help='maximum length of text')
    parser.add_argument('--d_model', type=int, default=768, required=False,
                        help='hidden size of the model')
    parser.add_argument('--heads', type=int, default=8, required=False,
                        help='number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, required=False,
                        help='dropout rate')

    parser.add_argument('--epochs', type=int, default=50, required=False,
                        help='number of epochs')
    parser.add_argument('--teacher_epochs', type=int, default=1, required=False,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0004, required=False,
                        help='learning rate')
    parser.add_argument('--early_stopping_rounds', type=int, default=10, required=False,
                        help='number of epochs for early stopping')
    parser.add_argument('--lr_schedule_round', type=int, default=30, required=False,
                        help='number of epochs for learning rate scheduling')

    parser.add_argument('--train_batch_size', type=int, default=4, required=False,
                        help='train batch size')
    parser.add_argument('--eval_batch_size', type=int, default=4, required=False,
                        help='eval batch size')

    parser.add_argument('--similarity_loss_weight', type=float, default=0.25, required=False,
                        help='weight for similarity loss')
    parser.add_argument('--similarity_topk', type=int, default=5, required=False,
                        help='Top k tokens for similarity loss computation')
    parser.add_argument('--encoder_kl_weight', type=float, default=0.5, required=False,
                        help='weight for encoder KL divergence loss')
    parser.add_argument('--distillation_kl_weight', type=float, default=0.5, required=False,
                        help='weight for KL divergence loss between student encoder and teacher')

    parser.add_argument('--model_save_path', type=str, default='../models/', required=False,
                        help='model save path')

    parser.add_argument('--wandb_logging', action='store_true',
                        help='wandb logging needed')
    parser.add_argument('--wandb_project_name', type=str, default='CodeMixed Generation', required=False,
                        help='wandb project name')

    parser.add_argument('--seed', type=int, default=42, required=False,
                        help='seed')


    args = parser.parse_args()
    print (args)

    df = pd.read_csv(args.data_file,sep='\t',lineterminator='\n').dropna(subset=['text']).reset_index(drop=True)
    df = df[df.base_language == 'hin']
    
    if 'author' not in df.columns:
        df['author'] = df['author_id'].copy()

    if 'cid' in df.columns:
        authors = df.groupby(['author'])['cid'].nunique().reset_index()
        authors = authors[authors.cid > 2][['author']]
    else:
        authors = df.groupby(['author'])['id'].nunique().reset_index()
        authors = authors[authors.id > 2][['author']]

    df = pd.merge(df, authors, how='inner')
    df['text'] = df.text.apply(lambda x: remove_emojis(remove_html(remove_email(clean_tweets(x)))).lower())
    df = df.sort_values(['author', 'posted_time_in_years'],ascending=[True, False]).reset_index(drop=True)
    #df = df.sample(frac=1).reset_index(drop=True)
    
    kf = StratifiedKFold(n_splits=2, shuffle=False)
    for train_index, test_index in kf.split(df.text, df.author):
        break

    train_df = df.iloc[train_index]
    val_df = df.iloc[test_index].reset_index(drop=True)

    kf2 = StratifiedKFold(n_splits=2, shuffle=True)
    for val_index, test_index in kf2.split(val_df.text, val_df.author):
        break

    test_df = val_df.iloc[test_index]
    val_df = val_df.iloc[val_index]

    #CMI_gt, M_index_gt, B_gt, LE_gt, _ = calculate_metrics(val_df.text.values)
    #print ("CMI:{}, M Index:{}, Burstiness:{}, Entropy:{}".format(CMI_gt, M_index_gt, B_gt, LE_gt))

    custom_wp_tokenizer(train_df.text.values, args.model_save_path, args.model_save_path, vocab_size=100000)
    custom_tokenizer = BertTokenizer.from_pretrained(args.model_save_path)
    bert_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    bert_model = AutoModel.from_pretrained(args.teacher_model)

    if args.use_distillation == True or args.use_meta_distillation == True:
        vocab_size = bert_tokenizer.vocab_size #bert_tokenizer.vocab_size
        trg_vocab_size = bert_tokenizer.vocab_size
    else:
        vocab_size = custom_tokenizer.vocab_size
        bert_tokenizer = custom_tokenizer
    
    trg_vocab_size = custom_tokenizer.vocab_size
    n_speaker = train_df.author.nunique() + 1
    d_model = bert_model.config.hidden_size
    max_len = args.max_text_len
    N = args.n_layers
    heads = 8

    print (vocab_size, trg_vocab_size, n_speaker, d_model, max_len, N, heads)

    ll = LabelEncoder()
    train_df['author_id'] = ll.fit_transform(train_df.author.values.reshape(-1,1))
    val_df['author_id'] = ll.transform(val_df.author.values.reshape(-1,1))
    test_df['author_id'] = ll.transform(test_df.author.values.reshape(-1,1))

    train_dataset = TransformerDataset(train_df.text.values.tolist(), train_df.author_id.values.tolist(), \
                                  src_tokenizer=bert_tokenizer, trg_tokenizer=custom_tokenizer, MAX_LEN=args.max_text_len)
    quiz_dataset = TransformerDataset(val_df.text.values.tolist(), val_df.author_id.values.tolist(), \
                                  src_tokenizer=bert_tokenizer, trg_tokenizer=custom_tokenizer, MAX_LEN=args.max_text_len)
    val_dataset = TransformerDataset(test_df.text.values.tolist(), test_df.author_id.values.tolist(), \
                                  src_tokenizer=bert_tokenizer, trg_tokenizer=custom_tokenizer, MAX_LEN=args.max_text_len)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=False)

    quiz_loader = torch.utils.data.DataLoader(
        quiz_dataset, batch_size=args.train_batch_size, shuffle=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False)

    if args.use_distillation == True:
        model = get_model(args, vocab_size, trg_vocab_size, speaker_count=n_speaker, \
                    use_persona=args.use_persona, generate_speaker_emb=args.generate_random_speaker, bert_model=bert_model,use_fame=args.use_fame, use_alignment=args.use_alignment,\
                        use_linear_speaker_emb=args.use_linear_speaker_emb, no_speaker_id=args.no_speaker_id)
    else:
        model = get_model(args, vocab_size, trg_vocab_size, speaker_count=n_speaker, \
                    use_persona=args.use_persona, generate_speaker_emb=args.generate_random_speaker, use_fame=args.use_fame, use_alignment=args.use_alignment, \
                        use_linear_speaker_emb=args.use_linear_speaker_emb, no_speaker_id=args.no_speaker_id)
    
    print ("Total number of parameters={}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    now = int(datetime.now().timestamp())
    data_name = os.path.basename(args.data_file).split('.')[0]

    if args.use_meta_distillation == True:
        model_checkpoint_name = "{}_meta_{}_{}.pth".format(model.__class__.__name__, data_name, str(now))
    else:
        model_checkpoint_name = "{}_{}_{}.pth".format(model.__class__.__name__, data_name, str(now))

    #model_checkpoint_name = "TransformerwithPersona_youtube_data_1703652493.pth"

    if args.wandb_logging == True:
        config = vars(args)
        config['model_name'] = model.__class__.__name__
        config['model_checkpoint'] = model_checkpoint_name
        wandb.login()
        wandb.init(project=args.wandb_project_name,config=config)
        artifact = wandb.Artifact('Model', type='model')
        wandb.watch(model, log_freq=100)
    
    #CMI, M_index, B, LE, repeat_index = calculate_metrics(test_df.text.values)

    #if args.wandb_logging == True:
    #    wandb.log({"text_CMI": test_df.CMI.astype(float).mean()})
    #    wandb.log({"text_M_index": M_index})
    #    wandb.log({"text_Burstiness": B})
    #    wandb.log({"text_Entropy": LE})

    if args.use_meta_distillation == True:
        best_validation_texts = meta_trainer.trainer(args, model, bert_model, train_loader, quiz_loader, val_loader, model_checkpoint_name)
    else:
        best_validation_texts = trainer_without_metalearning.trainer(args, model, train_loader, quiz_loader, val_loader, model_checkpoint_name)

    model = torch.load(os.path.join(args.model_save_path, model_checkpoint_name))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    
    all_preds = []

    for batch in tqdm(val_loader):
        src = batch['input_ids'].to(device)
        speaker_ids = batch['speaker_ids'].to(device)
        src2 = batch['decoder_input_ids'].to(device)

        decoder_src = src2[:,1:-1]
        target = src2[:,2:].contiguous().view(-1)

        src_mask, trg_mask = create_masks(src, decoder_src)

        with torch.no_grad():
            preds, _, _, _ = model(src, decoder_src, speaker_ids, src_mask, trg_mask)

        all_preds.append(preds.argmax(-1).detach().cpu().numpy())
    all_preds = np.concatenate(all_preds, 0)

    all_val_texts = []
    for i in range(all_preds.shape[0]):
        all_val_texts.append(custom_tokenizer.decode(all_preds[i]).replace('[CLS]','').replace('[SEP]','').replace('[PAD]','').strip())

    CMI, M_index, B, LE, repeat_index = calculate_metrics(all_val_texts)

    if args.wandb_logging == True:
        wandb.log({"validation_CMI": CMI})
        wandb.log({"validation_M_index": M_index})
        wandb.log({"validation_Burstiness": B})
        wandb.log({"validation_Entropy": LE})
    
    ####################### generation ######################
    val_df2 = val_df.drop_duplicates(subset=['author_id'])
    val_df2 = val_df2.rename(columns={'text':'old_text'})
    #best_validation_texts = np.concatenate(best_validation_texts,0)
    #best_validation_texts = [custom_tokenizer.decode(i).replace('[CLS]','').replace('[SEP]','').replace('[PAD]','').strip() for i in best_validation_texts]
    
    test_df2 = pd.merge(test_df, val_df2[['author_id','old_text']], how='left')[['author_id','text','old_text']] #.drop_duplicates(['author_id'])
    #test_df2['reconstructed_text'] = best_validation_texts
    test_df2 = test_df2.dropna().reset_index(drop=True)

    all_generated_texts = []
    random_generated_texts = []
    random_author_ids = []

    #args.max_text_len = 40 # remove later
    
    for i in tqdm(range(test_df2.shape[0])):
        all_generated_texts.append(generate_transformer(model, bert_tokenizer, custom_tokenizer, \
                                                    test_df2.old_text.iloc[i], " ".join(test_df2.text.iloc[i].split()[:1]), \
                                                    test_df2.author_id.iloc[i], args.max_text_len))
        random_author = random.sample(test_df2.author_id.unique().tolist(), 1)[0]
        random_generated_texts.append(generate_transformer(model, bert_tokenizer, custom_tokenizer, \
                                                    test_df2.old_text.iloc[i], " ".join(test_df2.text.iloc[i].split()[:1]), \
                                                    random_author, args.max_text_len))
        random_author_ids.append(random_author)

    test_df2['generated_text_coldstart'] = all_generated_texts
    test_df2['random_author_ids'] = random_author_ids
    test_df2['random_generated_texts_coldstart'] = random_generated_texts

    CMI, M_index, B, LE, repeat_index = calculate_metrics(test_df2.generated_text_coldstart.values)

    if args.wandb_logging == True:
        #wandb.log({"CS_test_CMI": CMI})
        wandb.log({"CS_test_M_index": M_index})
        wandb.log({"CS_test_Burstiness": B})
        wandb.log({"CS_test_Entropy": LE})

    all_generated_texts2 = []

    for i in tqdm(range(test_df2.shape[0])):
        all_generated_texts2.append(generate_transformer(model, bert_tokenizer, custom_tokenizer, \
                                                    test_df2.old_text.iloc[i], " ".join(test_df2.text.iloc[i].split()[:10]), \
                                                    test_df2.author_id.iloc[i], args.max_text_len*2))

    test_df2['generated_text'] = all_generated_texts2

    CMI, M_index, B, LE, repeat_index = calculate_metrics(test_df2.generated_text.values)

    if args.wandb_logging == True:
        #wandb.log({"test_CMI": CMI})
        wandb.log({"test_M_index": M_index})
        wandb.log({"test_Burstiness": B})
        wandb.log({"test_Entropy": LE})
    
    tokenizer_ = AutoTokenizer.from_pretrained("sagorsarker/codeswitch-hineng-lid-lince")
    model_ = AutoModelForTokenClassification.from_pretrained("sagorsarker/codeswitch-hineng-lid-lince")
    lid_model = pipeline('ner', model=model_, tokenizer=tokenizer_)
        
    CMI = []
    #CMI_old = []

    for i in tqdm(range(test_df2.shape[0])):
        CMI.append(calculate_CMI(test_df2.generated_text_coldstart.iloc[i], lid_model))
        #CMI_old.append(calculate_CMI(test_df2.text.iloc[i], lid_model))

    #test_df2['CMI_actual'] = CMI_old
    test_df2['CMI_generated'] = CMI

    #CMI = []
    #CMI_old = []

    #for i in tqdm(range(test_df2.shape[0])):
    #    CMI.append(calculate_CMI(test_df2.generated_text.iloc[i], lid_model))
    #    #CMI_old.append(calculate_CMI(test_df2.text.iloc[i], lid_model))

    #test_df2['CMI_actual'] = CMI_old
    #test_df2['CMI_generated_prompted'] = CMI

    if args.wandb_logging == True:
        wandb.log({"CS_test_CMI": test_df2['CMI_generated'].mean()})
        #wandb.log({"test_CMI": test_df2['CMI_generated_prompted'].mean()})

    tokenizer_ = AutoTokenizer.from_pretrained("sagorsarker/codeswitch-hineng-lid-lince")
    model_ = AutoModelForTokenClassification.from_pretrained("sagorsarker/codeswitch-hineng-lid-lince")

    #test_df2 = test_df2.dropna(subset=['old_text','text','generated_text','generated_text_coldstart']).reset_index(drop=True)
    test_df2 = test_df2.dropna(subset=['old_text','text','generated_text_coldstart']).reset_index(drop=True)

    rogue1 = []
    rogueL = []
    blues = []

    for i in range(test_df2.shape[0]):
        lid1 = language_identification(lid_model, test_df2.text.iloc[i])
        lid2 = language_identification(lid_model, test_df2.generated_text_coldstart.iloc[i])

        lid1 = " ".join(list(lid1.values()))
        lid2 = " ".join(list(lid2.values()))

        scorer = rouge_scorer.RougeScorer(['rouge1','rougeL'], use_stemmer=True)
        scores = scorer.score(lid1, lid2)

        rogue1.append(scores['rouge1'].fmeasure)
        rogueL.append(scores['rougeL'].fmeasure)

        try:
            blues.append(sentence_bleu([lid1.split()], lid2.split()))
        except:
            blues.append(0)

    test_df2['rouge1_cs'] = rogue1
    test_df2['rougeL_cs'] = rogueL
    test_df2['bleu_cs'] = blues

    
    rogue1 = []
    rogueL = []
    blues = []

    for i in range(test_df2.shape[0]):
        lid1 = language_identification(lid_model, test_df2.text.iloc[i])
        lid2 = language_identification(lid_model, test_df2.generated_text.iloc[i])

        lid1 = " ".join(list(lid1.values()))
        lid2 = " ".join(list(lid2.values()))

        scorer = rouge_scorer.RougeScorer(['rouge1','rougeL'], use_stemmer=True)
        scores = scorer.score(lid1, lid2)

        rogue1.append(scores['rouge1'].fmeasure)
        rogueL.append(scores['rougeL'].fmeasure)

        try:
            blues.append(sentence_bleu([lid1.split()], lid2.split()))
        except:
            blues.append(0)

    test_df2['rouge1'] = rogue1
    test_df2['rougeL'] = rogueL
    test_df2['bleu'] = blues

    if args.wandb_logging == True:
        wandb.log({"CS_rogue1": test_df2['rouge1_cs'].mean()})
        wandb.log({"CS_rogueL": test_df2['rougeL_cs'].mean()})
        wandb.log({"CS_bleu": test_df2['bleu_cs'].mean()})
        #wandb.log({"rogue1": test_df2['rouge1'].mean()})
        #wandb.log({"rogueL": test_df2['rougeL'].mean()})
        #wandb.log({"bleu": test_df2['bleu'].mean()})

    test_df2.to_csv(os.path.join(args.model_save_path, 'generated_texts_{}.csv'.format(now)), sep='\t', index=False)
    if args.wandb_logging == True:
        wandb.log({"generated_text": os.path.join(args.model_save_path, 'generated_texts_{}.csv'.format(now))})
