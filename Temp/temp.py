import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np

import utils

import time

# Dataset
train_file='data/yahoo/yahoo-train.hdf5'
val_file='data/yahoo/yahoo-val.hdf5'
test_file='data/yahoo/yahoo-test.hdf5'
from data import Dataset
train_data=Dataset(train_file)
val_data=Dataset(val_file)
vocab_size=train_data.vocab_size

# Model
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden_to_mean = nn.Linear(hidden_dim, latent_dim)
        self.hidden_to_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = hidden.squeeze(0)
        mean = self.hidden_to_mean(hidden)
        logvar = self.hidden_to_logvar(hidden)
        return mean, logvar

class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, latent_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim + latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, hidden, z):
        embedded = self.embedding(x)
        z_repeated = z.unsqueeze(1).repeat(1, embedded.size(1), 1)
        lstm_input = torch.cat([embedded, z_repeated], dim=2)
        output, hidden = self.lstm(lstm_input, hidden)
        prediction = self.fc_out(output)
        return prediction, hidden

class VAE(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, embedding_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(output_dim, embedding_dim, hidden_dim, latent_dim)

    def forward(self, x, target, teacher_forcing_ratio=0.5):
        mean, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)
        hidden = (z.unsqueeze(0), torch.zeros_like(z).unsqueeze(0))
        
        decoder_input = target[:, 0].unsqueeze(1)
        predictions = []

        for t in range(1, target.size(1)):
            output, hidden = self.decoder(decoder_input, hidden, z)
            predictions.append(output)

            top1 = output.argmax(2)
            decoder_input = target[:, t].unsqueeze(1) if torch.rand(1).item() < teacher_forcing_ratio else top1

        return torch.cat(predictions, dim=1), mean, logvar

def loss_function(recon_x, x, mean, logvar):
    recon_loss = F.cross_entropy(recon_x.transpose(1, 2), x, reduction='mean')
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kl_loss

# Model Parameters
INPUT_DIM = OUTPUT_DIM = vocab_size  # Example vocabulary size
EMBEDDING_DIM = 512
# HIDDEN_DIM = 1024
HIDDEN_DIM = 32

LATENT_DIM = 32
DROPOUT = 0.5
BATCH_SIZE = 32
LEARNING_RATE = 1.0
MAX_EPOCHS = 120
KL_ANNEAL_EPOCHS = 10
DECAY_FACTOR = 0.5
PATIENCE = 5
BETA=0.1




device=torch.device("cuda" if torch.cuda.is_available() else "cpu")




# Model and Optimizer
model = VAE(INPUT_DIM, OUTPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, LATENT_DIM).to(device)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Training Loop
kl_weight = 0.0
best_loss = float('inf')
no_improve_epochs = 0

t = 0
best_val_nll = 1e5
best_epoch = 0
val_stats = []
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
    

    for i in np.random.permutation(len(train_data)):
        # Data
        sentences, length, batch_size = train_data[i]
        sentences=sentences.to(device)
        
        # MODEL
        ## Encoder forward
        mean, logvar = model.encoder(sentences)
        ## Reparameterization trick
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)
        ## Decoder forward
        hidden = (z.unsqueeze(0), torch.zeros_like(z).unsqueeze(0))
        predictions,hidden =model.decoder(sentences,hidden, z)
        print(predictions.shape)
        print(sentences.shape)
        
        # Loss
        ## NLL Loss
        nll_vae=sum([nn.NLLLoss(predictions[:, l], sentences[:, l+1]) for l in range(length)])
        train_nll_vae+=nll_vae.item()*batch_size
        ## KL Loss
        kl_vae = utils.kl_loss_diag(mean, logvar)
        train_kl_vae += kl_vae.data*batch_size        
        ## VAE Loss
        vae_loss = nll_vae + BETA*kl_vae          
        vae_loss.backward(retain_graph = True)
        
        optimizer.step()
        num_sents += batch_size
        num_words += batch_size * length
        
        b += 1
        if b % 100 == 0:
          param_norm = sum([p.norm()**2 for p in model.parameters()]).data[0]**0.5
          
          print('Iters: %d, Epoch: %d, Batch: %d/%d, LR: %.4f,  TrainVAE_PPL: %.2f, TrainVAE_KL: %.4f, TrainVAE_PPLBnd: %.2f,  KLInitFinal: %.2f, |Param|: %.4f, BestValPerf: %.2f, BestEpoch: %d, Beta: %.4f, Throughput: %.2f examples/sec' % 
                (t, 
                 epoch, 
                 b+1, len(train_data), 
                 LEARNING_RATE, 
                 np.exp(train_nll_vae/num_words), 
                 train_kl_vae / num_sents,
                 np.exp((train_nll_vae + train_kl_vae)/num_words),
                 param_norm, best_val_nll, best_epoch, BETA,
                 num_sents / (time.time() - start_time)))


        
        
    
    # for x, target in train_loader:
    #     x, target = x.to(model.device), target.to(model.device)
    #     optimizer.zero_grad()

    #     recon_x, mean, logvar = model(x, target)
    #     loss = loss_function(recon_x, target, mean, logvar) + kl_weight
    #     loss.backward()
    #     optimizer.step()

    #     epoch_loss += loss.item()

    # epoch_loss /= len(train_loader)
    # print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

    # if epoch_loss < best_loss:
    #     best_loss = epoch_loss
    #     no_improve_epochs = 0
    # else:
    #     no_improve_epochs += 1

    # if no_improve_epochs >= PATIENCE:
    #     LEARNING_RATE *= DECAY_FACTOR
    #     optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    #     no_improve_epochs = 0

    # if LEARNING_RATE < 1e-5:
    #     print("Early stopping due to learning rate decay.")
    #     break
