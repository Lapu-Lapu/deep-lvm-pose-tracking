from tqdm import tqdm_notebook as tqdm

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch import Tensor as T

from contextlib import ExitStack
import numpy as np
from math import pi


def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std


class cVAE(nn.Module):
    def __init__(self, input_dim, cond_data_len, latent_dim=5, hidden=40):
        nn.Module.__init__(self)
        self.enc = nn.ModuleList([
            # nn.Linear(input_dim, hidden),
            # nn.Linear(hidden, 2 * latent_dim)
            nn.Linear(input_dim, 2 * latent_dim)
        ])
        self.enc_activations = [
            # F.relu,
            lambda x: x
        ]
        self.dec = nn.ModuleList([
            nn.Linear(latent_dim + cond_data_len, hidden),
            nn.Linear(hidden, input_dim)
        ])
        self.dec_activations = [
            F.relu,
            torch.sigmoid
        ]
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-4)
        self.latent_dim = latent_dim
        self.cond_data_len = cond_data_len

    def encode(self, x):
        out = x
        # import pdb; pdb.set_trace()
        for layer, activation in zip(self.enc, self.enc_activations):
            out = activation(layer(out))
        return out[:, :self.latent_dim], out[:, self.latent_dim:]

    def decode(self, z):
        out = z
        for layer, activation in zip(self.dec, self.dec_activations):
            out = activation(layer(out))
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        z_cond = torch.cat((z, x[:, -self.cond_data_len:]), dim=1)
        return self.decode(z_cond), mu, logvar


class VAE(cVAE):
    def __init__(self, input_dim, latent_dim=5, hidden=40):
        cVAE.__init__(self, input_dim, latent_dim=latent_dim,
                      hidden=hidden, cond_data_len=0)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def to(self, device):
        self.device = device
        return super().to(device)


def loss_function(recon_x, x, mu, logvar, beta=1, likelihood='mse'):
    """
    Reconstruction + KL divergence losses summed over all elements
    and averaged over batch
    recon_x: reconstructed input (batch_size, dimension)
    x: input (batch_size, dimension)
    mu: \mu(x) - mean of q(z|x, \phi) (batch_size, latent_dim)
    logvar: \log(\sigma^2) of latent variable of input (batch_size, latent_dim)
    """
    M, D = recon_x.shape
    if likelihood == 'bce':  ## Binary cross entropy
        # x_hat \log(x) + (1-x_hat) \log(1-x)
        rec_err = F.binary_cross_entropy(recon_x, x, reduction='sum')
    elif likelihood == 'mse': ## Mean squared error
        rec_err = 0.5 * torch.mean((recon_x - x)**2)  # + 0.5*M*D*np.log(0.5*pi)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    KLD = torch.mean(KLD)
    return rec_err + beta * KLD


class convVAE(nn.Module):
    def __init__(self, latent_dim=7):
        """
        Work in progress
        """
        nn.Module.__init__(self)

        self.encoder = [
            # input (b, 1, 64, 64 (formerly 28))
            nn.Conv2d(1, 1, 3, stride=3, padding=1),  # b, 1, 22
            nn.MaxPool2d(2, stride=2),  # b, 1, 11
            nn.Conv2d(1, 1, 3, stride=1, padding=1),  # b, 1, 11
            nn.MaxPool2d(2, stride=1)  # b, 1, 10
        ]
        # flatten: (1, 10, 10) -> 100
        self.to_mu = nn.Linear(80, latent_dim)
        self.to_sig = nn.Linear(latent_dim, 80)
        # unflatten : 100 -> (1, 10, 10)
        self.decoder = [
            F.interpolate(size=(20, 20)),
            F.interpolate(size=(20, 20)),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=1, padding=1),  # b, 1, 28, 28
            nn.Sigmoid()
        ]

    def encode(self, x):
        h1 = F.relu(self.encoder[0](x))
        h2 = self.encoder[1](h1)
        h3 = F.relu(self.encoder[2](h2))
        return self.encoder[3](h3)

    def decode(self, z):
        pass


def fit(model, data_loader, epochs=5, verbose=True, optimizer=None,
        device='cpu', weight_fn=None, conditional=False,
        loss_func=None):
    """
    model: instance of nn.Module
    data_loader: Dictionary of pytorch.util.data.DataSet for training and
                 validation
    """
    if conditional:
        likelihood = 'mse'
    else:
        likelihood = 'bce'
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    if loss_func is None:
        loss_func = loss_function

    all_train_loss = []
    for epoch in range(epochs):
        for phase in ['train', 'val']:
            epoch_loss = []
            N = len(data_loader[phase].dataset)
            M = data_loader[phase].batch_size
            with ExitStack() as stack:
                if phase == 'train':
                    model.train()
                    pbar = stack.enter_context(tqdm(data_loader[phase]))
                else:
                    model.eval()
                    stack.enter_context(torch.no_grad())
                    pbar = stack.enter_context(tqdm(data_loader[phase], disable=True))

                batch_idx = 0
                for batch in pbar:
                    batch_idx += 1
                    img = batch['image'].view(data_loader[phase].batch_size, -1)
                    if conditional:
                        label = batch['angles']
                        batch = torch.cat((img, label), dim=1).float()
                        data = T(batch.float()).to(device)
                    else:
                        data = T(img.float()).to(device)
                    recon_batch, mu, logvar = model(data)
                    loss = loss_func(recon_x=recon_batch, x=data, mu=mu, logvar=logvar)
                    optimizer.zero_grad()
                    if phase == 'train':
                        loss.backward()
                        epoch_loss += [loss.item()]
                        # grad = loss.grad
                        # import pdb; pdb.set_trace()
                        optimizer.step()
                    else:
                        epoch_loss += [loss.item()]
                    if verbose and phase == 'train':
                        pbar.set_description(
                            f"({phase}) Epoch {epoch}, Loss at batch {batch_idx:05d}: {loss.item()*M/N:.2E}"
                        )
            if phase == 'train':
                all_train_loss += epoch_loss
                epoch_loss = loss.item()
            else:
                epoch_loss = np.sum(np.array(epoch_loss))
            print(f'{phase} loss: {epoch_loss*M/N:.2E}')
            if weight_fn is not None:
                torch.save(model.state_dict(), weight_fn)
    return all_train_loss

if __name__ == '__main__':
    cVAE(1, 1)
