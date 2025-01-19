import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import scale_utils.kl_loss as kl_loss




def eval(data, model,device):
    model.eval()
    criterion = nn.NLLLoss()
    num_sentences = 0
    num_words = 0
    total_nll_vae = 0.
    total_kl_vae = 0.
    
    for i in range(len(data)):
        # Data
        sentences, length, batch_size = data[i]
        sentences=sentences.to(device)
        
        num_words += batch_size*length
        num_sentences += batch_size
  
        # MODEL
        ## Encoder forward
        mean, logvar = model.encoder_forward(sentences)
        ## Reparameterization trick
        z = model.reparameterize(mean, logvar)
        ## Decoder forward
        predictions = model.decoder_forward(sentences, z)
        
        # LOSS
        ## NLL Loss
        nll_vae=sum([criterion(predictions[:, l], sentences[:, l+1]) for l in range(length)])
        total_nll_vae+=nll_vae.item()*batch_size
        ## KL Loss
        kl_vae = kl_loss.kl_loss_diag(mean, logvar)
        total_kl_vae += kl_vae.data*batch_size
    
    ppl_vae = torch.exp(total_nll_vae/ num_words)
    kl_vae = total_kl_vae / num_sentences
    ppl_bound_vae = torch.exp((total_nll_vae + total_kl_vae)/num_words)

    print('VAE PPL: %.4f, VAE KL: %.4f, VAE PPLBnd: %.4f' % (ppl_vae, kl_vae, ppl_bound_vae))
    
    return ppl_bound_vae