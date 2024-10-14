import os
import sys
from tqdm import tqdm
from collections import OrderedDict
from copy import deepcopy as cp
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

sys.path.append("..")

from models.utils import create_masks

def return_weights(args, student_weights, grads):
  out = {}

  for (name, param), grad in zip(student_weights.items(), grads):
    if grad is None:
      out[name] = param
    else:
      out[name] = param - args.lr * grad

  return OrderedDict(out)

def trainer(args, model, bert_model, train_loader, quiz_loader, val_loader, model_checkpoint_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    bert_model = bert_model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    #model_checkpoint_name = "{}.pth".format(model.__class__.__name__)

    max_grad_norm = 5
    best_val_loss = 9999
    bad_epochs = 0

    optimizer1 = torch.optim.SGD(model.parameters(), lr=args.lr)
    optimizer2 = torch.optim.SGD(bert_model.parameters(), lr=args.lr/10)
    optimizer3 = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    for epoch in range(args.epochs):
        model.train()
        bert_model.eval()

        #student_weights = OrderedDict((name, param) for (name, param) in model.named_parameters())
        #s_model_backup_state_dict, s_optimizer_backup_state_dict = cp(model.state_dict()), cp(
        #                    optimizer1.state_dict())
                    
        if bad_epochs < args.early_stopping_rounds:

            # Meta learning on student model
            for i, batch in tqdm(enumerate(train_loader)):
                src = batch['input_ids'].to(device)
                speaker_ids = batch['speaker_ids'].to(device)
                src2 = batch['decoder_input_ids'].to(device)

                decoder_src = src2[:,1:-1]
                target = src2[:,2:].contiguous().view(-1)

                src_mask, trg_mask = create_masks(src, decoder_src)

                #output, proj_matrix, e_output = model(src, decoder_src, speaker_ids, speaker_mask=1, src_mask=src_mask, trg_mask=trg_mask)
                optimizer1.zero_grad()

                preds, kl_loss, e_outputs, proj_matrix = model(src, decoder_src, speaker_ids, src_mask, trg_mask)

                e_output = e_outputs[:,0,:]
            
                with torch.no_grad():
                    bert_output = bert_model(src)[1]
            
                #kl_loss = nn.KLDivLoss(reduction='mean')(e_output.view(-1,e_output.size()[-1]), bert_output.view(-1,bert_output.size()[-1]))
                kl_loss = nn.MSELoss()(e_output, bert_output)

                total_loss = F.cross_entropy(preds.view(-1, preds.size(-1)), target, ignore_index=0)
                ce_loss = total_loss.clone()
                #total_loss += args.encoder_kl_weight * (kl_loss + model.kl)
                total_loss += args.encoder_kl_weight * model.kl

                similarity_loss = 0

                #for bs in range(preds.size()[0]):
                #    #similarity_loss -= torch.log(torch.topk(nn.Softmax()(preds[bs])[:,target[bs]], args.similarity_topk).values.prod(-1)**(1/args.similarity_topk)).mean()
                #    top_tokens = torch.topk(proj_matrix[:,target[bs]], args.similarity_topk).indices
                #    similarity_loss -= torch.log(nn.Softmax()(preds[bs])[:,top_tokens].prod(-1)**(1/args.similarity_topk)).mean()

                similarity_loss /= preds.size()[0]
                total_loss += args.similarity_loss_weight * similarity_loss

                #grads = torch.autograd.grad(total_loss, model.parameters() if i == 0 else student_weights.values(),
                #                                            create_graph=True, retain_graph=True, allow_unused=True)

                #student_weights = return_weights(args, student_weights, grads)
                total_loss.backward()
                optimizer1.step()

                #if i%100 == 0:
                #  print ("Current training CE loss {}".format(total_loss.item()))

            # Meta learning on teacher model
            #student_prime_model = cp(model)
            #student_prime_model.load_state_dict(student_weights)
            #model.load_state_dict(student_weights)

            model.eval()
            bert_model.train()

            for batch in tqdm(quiz_loader):
                src = batch['input_ids'].to(device)
                speaker_ids = batch['speaker_ids'].to(device)
                src2 = batch['decoder_input_ids'].to(device)

                decoder_src = src2[:,1:-1]
                target = src2[:,2:].contiguous().view(-1)

                src_mask, trg_mask = create_masks(src, decoder_src)

                #output, proj_matrix, e_output = model(src, decoder_src, speaker_ids, speaker_mask=1, src_mask=src_mask, trg_mask=trg_mask)
                optimizer2.zero_grad()

                with torch.no_grad():
                    preds, kl_loss, e_outputs, proj_matrix = model(src, decoder_src, speaker_ids, src_mask, trg_mask)
            
                e_output = e_outputs[:,0,:]
            
                bert_output = bert_model(src)[1]
            
                #kl_loss = nn.KLDivLoss(reduction='mean')(e_output.view(-1,e_output.size()[-1]), bert_output.view(-1,bert_output.size()[-1]))
                kl_loss = nn.MSELoss()(e_output, bert_output)

                #total_loss = F.cross_entropy(preds.view(-1, preds.size(-1)), target, ignore_index=0)
                #ce_loss = total_loss.clone()
                total_loss = kl_loss
                #total_loss += args.encoder_kl_weight * kl_loss

                similarity_loss = 0

                #for bs in range(preds.size()[0]):
                #    #similarity_loss -= torch.log(torch.topk(nn.Softmax()(preds[bs])[:,target[bs]], args.similarity_topk).values.prod(-1)**(1/args.similarity_topk)).mean()
                #    top_tokens = torch.topk(proj_matrix[:,target[bs]], args.similarity_topk).indices
                #    similarity_loss -= torch.log(nn.Softmax()(preds[bs])[:,top_tokens].prod(-1)**(1/args.similarity_topk)).mean()

                similarity_loss /= preds.size()[0]
                #total_loss += args.similarity_loss_weight * similarity_loss

                #t_grads = torch.autograd.grad(total_loss, bert_model.parameters())

                #for p, gr in zip(bert_model.parameters(), t_grads):
                #    p.grad = gr
                total_loss.backward()

                #torch.nn.utils.clip_grad_norm_(bert_model.parameters(), max_grad_norm)

                optimizer2.step()

                # Manual zero_grad
                for p in bert_model.parameters():
                    p.grad = None

            # Learning of student model
            #model.load_state_dict(s_model_backup_state_dict)
            #optimizer1.load_state_dict(s_optimizer_backup_state_dict)
            
            #del student_weights, s_model_backup_state_dict, s_optimizer_backup_state_dict, grads
            
            model.train()
            bert_model.eval()

            training_loss = 0
            training_ce_loss = 0

            for i, batch in tqdm(enumerate(train_loader)):
                src = batch['input_ids'].to(device)
                speaker_ids = batch['speaker_ids'].to(device)
                src2 = batch['decoder_input_ids'].to(device)

                decoder_src = src2[:,1:-1]
                target = src2[:,2:].contiguous().view(-1)

                src_mask, trg_mask = create_masks(src, decoder_src)

                optimizer3.zero_grad()

                #output, proj_matrix, e_output = model(src, decoder_src, speaker_ids, speaker_mask=1, src_mask=src_mask, trg_mask=trg_mask)
            
                preds, kl_loss, e_outputs, proj_matrix = model(src, decoder_src, speaker_ids, src_mask, trg_mask)

                e_output = e_outputs[:,0,:]
            
                with torch.no_grad():
                    bert_output = bert_model(src)[1]
            
                #kl_loss = nn.KLDivLoss(reduction='mean')(e_output.view(-1,e_output.size()[-1]), bert_output.view(-1,bert_output.size()[-1]))
                kl_loss = nn.MSELoss()(e_output, bert_output)

                total_loss = F.cross_entropy(preds.view(-1, preds.size(-1)), target, ignore_index=0)
                ce_loss = total_loss.clone()

                total_loss += args.encoder_kl_weight * (kl_loss + model.kl) #args.encoder_kl_weight * kl_loss
                
                similarity_loss = 0

                #for bs in range(preds.size()[0]):
                #    #similarity_loss -= torch.log(torch.topk(nn.Softmax()(preds[bs])[:,target[bs]], args.similarity_topk).values.prod(-1)**(1/args.similarity_topk)).mean()
                #    top_tokens = torch.topk(proj_matrix[:,target[bs]], args.similarity_topk).indices
                #    similarity_loss -= torch.log(nn.Softmax()(preds[bs])[:,top_tokens].prod(-1)**(1/args.similarity_topk)).mean()

                similarity_loss /= preds.size()[0]
                total_loss += args.similarity_loss_weight * similarity_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer3.step()

                training_loss += total_loss.item()
                training_ce_loss += ce_loss.item()

                #if i%100 == 0:
                #  print ("Current training CE loss {}".format(total_loss.item()))

            ################################## validation ##################################
            ################################################################################
            training_loss /= len(train_loader)
            training_ce_loss /= len(train_loader)

            model.eval()

            validation_loss = 0

            validation_texts = []

            for batch in val_loader:
                src = batch['input_ids'].to(device)
                speaker_ids = batch['speaker_ids'].to(device)
                src2 = batch['decoder_input_ids'].to(device)

                decoder_src = src2[:,1:-1]
                target = src2[:,2:].contiguous().view(-1)

                src_mask, trg_mask = create_masks(src, decoder_src)

                #with torch.no_grad():
                #    output, proj_matrix, e_output = model(src, decoder_src, speaker_ids, speaker_mask=1, src_mask=src_mask, trg_mask=trg_mask)

                #total_loss, ce_loss = loss(model.kl, output, target, proj_matrix)
                with torch.no_grad():
                    preds, _, _, _ = model(src, decoder_src, speaker_ids, src_mask, trg_mask)
            
                total_loss = F.cross_entropy(preds.view(-1, preds.size(-1)), target, ignore_index=0)
                validation_texts.append(preds.detach().cpu().numpy().argmax(-1))
                validation_loss += total_loss.item()
                #validation_ce_loss += ce_loss.item()

            validation_loss /= len(val_loader)
            #validation_ce_loss /= len(val_loader)

            ppl = math.exp(validation_loss)

            #print ("Epoch:{}, Train loss:{}, Train CrossEntropy loss:{}, Validation loss:{}, \
            #      Validation CrossEntropy loss:{}, Validation Perplexity:{}".format(epoch+1, training_loss, \
            #                                  training_ce_loss, validation_loss, validation_ce_loss, ppl))
        
            print ("Epoch:{}, Train loss:{}, Train ce loss:{}, Validation loss:{}, Validation Perplexity:{}".format(epoch+1, training_loss, training_ce_loss, \
                                            validation_loss, ppl))
        
            if validation_loss < best_val_loss:
                #torch.save(model.state_dict(), os.path.join(args.model_save_path, model_checkpoint_name))
                torch.save(model, os.path.join(args.model_save_path, model_checkpoint_name))
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
                wandb.log({"training_ce_loss": training_ce_loss})
                wandb.log({"validation_loss": validation_loss})
                wandb.log({"validation_perplexity": ppl})

    return best_validation_texts