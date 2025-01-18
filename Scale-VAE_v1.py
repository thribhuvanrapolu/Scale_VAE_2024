import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np


import time


from scale_utils.data import Dataset
from scale_utils.model import RNNVAE
import scale_utils.kl_loss as kl_loss
from eval_scale_vae_v1 import eval



# Dataset
train_file='datasets/yahoo/yahoo-train.hdf5'
val_file='datasets/yahoo/yahoo-val.hdf5'
test_file='datasets/yahoo/yahoo-test.hdf5'
train_data=Dataset(train_file)
val_data=Dataset(val_file)
vocab_size=train_data.vocab_size


# Model Parameters
INPUT_DIM = OUTPUT_DIM = vocab_size  # Example vocabulary size
EMBEDDING_DIM = 512
HIDDEN_DIM = 1024
LATENT_DIM = 32
DROPOUT = 0.5
BATCH_SIZE = 32
LEARNING_RATE = 1.0
MAX_EPOCHS = 120
KL_ANNEAL_EPOCHS = 10
DECAY_FACTOR = 0.5
PATIENCE = 5
BETA=0.1
MAX_GRAD_NORM=5
CHECKPOINT_PATH='baseline.pt'
PRINT_EVERY=100 
MAX_LR_DECAY_ITER=5

# SEED
SEED=3435
np.random.seed(SEED)
torch.manual_seed(SEED)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


# MODEL
model = RNNVAE( vocab_size = vocab_size,
                enc_word_dim = EMBEDDING_DIM,
                enc_h_dim = HIDDEN_DIM,
                enc_num_layers = 1,
                dec_word_dim = EMBEDDING_DIM,
                dec_h_dim = HIDDEN_DIM,
                dec_num_layers = 1,
                dec_dropout = DROPOUT,
                latent_dim = LATENT_DIM).to(device)


# for param in model.parameters():    
#     param.data.uniform_(-0.1, 0.1)


# Optimizer
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Criterion
criterion = nn.NLLLoss()


# TRAINING
kl_weight = 0.0
best_loss = float('inf')
no_improve_epochs = 0

t = 0
best_val_nll = 1e5
best_epoch = 0
val_stats = []

des_std=1.1
f_epo=1

iter_learning_rate_decay=0
iter=0

for epoch in range(MAX_EPOCHS):
    start_time = time.time()
    
    model.train()
    epoch_loss = 0
    kl_weight = min(1.0, epoch / KL_ANNEAL_EPOCHS)
    
    train_nll_vae = 0.
    train_kl_vae = 0.
    num_sents = 0
    num_words = 0
    b = 0
    f_list=[]
    # line 4
    for i in np.random.permutation(len(train_data)):
        # Data
        sentences, length, batch_size = train_data[i]
        sentences=sentences.to(device)
        
        # Optimizer
        optimizer.zero_grad()
        
        # MODEL
        ## Encoder forward (Line 5)
        mean, logvar = model.encoder_forward(sentences)
        # Line 6
        f=des_std/torch.std(mean)
        f_list.append(f)
        # Line 7-11
        if iter <= f_epo:
            scaled_mean = f * mean
        else:
            scaled_mean = scaled_f * mean
        
        ## Reparameterization trick (Line 12)
        z = model.reparameterize(scaled_mean, logvar)
        ## Decoder forward (Line 13)
        predictions = model.decoder_forward(sentences, z)
        
        
        # LOSS
        ## NLL Loss
        nll_vae=sum([criterion(predictions[:, l], sentences[:, l+1]) for l in range(length)])
        train_nll_vae+=nll_vae.item()*batch_size
        ## KL Loss
        kl_vae = kl_loss.kl_loss_diag(mean, logvar)
        train_kl_vae += kl_vae.data*batch_size        
        ## VAE Loss
        loss_scale = nll_vae + kl_weight*kl_vae          
        loss_scale.backward(retain_graph = True)
        
        
        if MAX_GRAD_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)        
        optimizer.step()
        num_sents += batch_size
        num_words += batch_size * length
        
        b += 1
        if b % PRINT_EVERY == 0:
            param_norm = sum([p.norm()**2 for p in model.parameters()]).data**0.5
            print('Epoch: %d, Batch: %d/%d, LR: %.4f,  TrainVAE_PPL: %.2f, TrainVAE_KL: %.4f, TrainVAE_PPLBnd: %.2f, |Param|: %.4f, BestValPerf: %.2f, BestEpoch: %d, Beta: %.4f, Throughput: %.2f examples/sec' % 
                  (epoch, 
                   b+1, len(train_data), 
                   LEARNING_RATE, 
                   torch.exp(train_nll_vae/num_words), 
                   train_kl_vae / num_sents,
                   torch.exp((train_nll_vae + train_kl_vae)/num_words),
                   param_norm, best_val_nll, best_epoch, BETA,
                   num_sents / (time.time() - start_time)))

    # Line 17 
    stacked_f=torch.stack(f_list)
    scaled_f = torch.mean(stacked_f,dim=0)
    # Line 18
    iter+=1
    
    print('--------------------------------')
    print('Checking validation perf...')
    val_nll = eval(val_data, model,device)
    val_stats.append(val_nll)
    if val_nll < best_val_nll:
      best_val_nll = val_nll
      best_epoch = epoch
      model.cpu()
      checkpoint = {
        # 'args': args.__dict__,
        'model': model,
        'val_stats': val_stats
      }
      print('NLL Loss for Validation: ',val_nll)
      print('Saving checkpoint to %s' % CHECKPOINT_PATH)      
      torch.save(checkpoint, CHECKPOINT_PATH)
      model.cuda()
    print('--------------------------------')

    if epoch - best_epoch >= 5:
        LEARNING_RATE /= DECAY_FACTOR
        iter_learning_rate_decay+=1
    
    if iter_learning_rate_decay==MAX_LR_DECAY_ITER:
        break