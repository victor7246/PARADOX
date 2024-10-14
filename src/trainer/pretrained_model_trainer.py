import os
import sys
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import bitsandbytes as bnb

sys.path.append("..")

from models.utils import create_masks

def trainer(args, model, train_loader, quiz_loader, val_loader, model_checkpoint_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = model.to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.lr)

    #model_checkpoint_name = "{}.pth".format(model.__class__.__name__)

    max_grad_norm = 1
    best_val_loss = 9999
    bad_epochs = 0

    with torch.cuda.amp.autocast():
        for epoch in range(args.epochs):
            if bad_epochs < args.early_stopping_rounds:
                training_loss = 0
                training_ce_loss = 0

                for i, batch in tqdm(enumerate(train_loader)):
                    src = batch['input_ids'].to(device)

                    optimizer.zero_grad()

                    #output, proj_matrix, e_output = model(src, decoder_src, speaker_ids, speaker_mask=1, src_mask=src_mask, trg_mask=trg_mask)
                
                    outputs = model(input_ids=src, labels=src)
                    total_loss, preds = outputs.loss, outputs.logits

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    optimizer.step()

                    training_loss += total_loss.item()

                for batch in tqdm(quiz_loader):
                    src = batch['input_ids'].to(device)

                    optimizer.zero_grad()

                    outputs = model(input_ids=src, labels=src)
                    total_loss, preds = outputs.loss, outputs.logits

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    optimizer.step()

                    training_loss += total_loss.item()

                training_loss /= (len(train_loader) + len(quiz_loader))

                ################################## validation ##################################
                ################################################################################

                model.eval()
                validation_texts = []
                validation_loss = 0

                for batch in val_loader:
                    src = batch['input_ids'].to(device)

                    with torch.no_grad():
                        outputs = model(input_ids=src, labels=src)

                    total_loss, preds = outputs.loss, outputs.logits
                    validation_texts.append(preds.detach().cpu().numpy().argmax(-1))

                    validation_loss += total_loss.item()
                    #validation_ce_loss += ce_loss.item()

                validation_loss /= len(val_loader)
                #validation_ce_loss /= len(val_loader)

                ppl = math.exp(validation_loss)

                print ("Epoch:{}, Train loss:{} Validation loss:{}, Validation Perplexity:{}".format(epoch+1, training_loss, \
                                                validation_loss, ppl))
            
                if validation_loss < best_val_loss:
                    #torch.save(model.state_dict(), os.path.join(args.model_save_path, model_checkpoint_name))
                    #torch.save(model, os.path.join(args.model_save_path, model_checkpoint_name))
                    best_val_loss = validation_loss
                    #artifact.add_file(os.path.join(args.model_save_path, model_checkpoint_name), name='Trained student model')
                    best_validation_texts = validation_texts

                    bad_epochs = 0
                    if args.wandb_logging:
                        #wandb.log({"best_training_loss": training_loss})
                        wandb.log({"best_validation_loss": validation_loss})
                        wandb.log({"best_validation_perplexity": ppl})
                else:
                    bad_epochs += 1

                if args.wandb_logging:
                    wandb.log({"training_loss": training_loss})
                    wandb.log({"validation_loss": validation_loss})
                    wandb.log({"validation_perplexity": ppl})
    
    return model, best_validation_texts