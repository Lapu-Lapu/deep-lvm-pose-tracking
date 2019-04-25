from tqdm import tqdm_notebook as tqdm

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch import Tensor as T

from contextlib import ExitStack
import numpy as np


def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std


class cVAE(nn.Module):
    def __init__(self, input_dim, cond_data_len, bottleneck=5, hidden=40):
        nn.Module.__init__(self)
        self.enc = nn.ModuleList([
            nn.Linear(input_dim, hidden),
            nn.Linear(hidden, 2 * bottleneck)
        ])
        self.dec = nn.ModuleList([
            nn.Linear(bottleneck + cond_data_len, hidden),
            nn.Linear(hidden, input_dim)
        ])
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-4)
        self.bottleneck = bottleneck
        self.cond_data_len = cond_data_len

    def encode(self, x):
        h = F.relu(self.enc[0](x))
        out = self.enc[1](h)
        return out[:, :self.bottleneck], out[:, self.bottleneck:]

    def decode(self, z):
        h = F.relu(self.dec[0](z))
        return torch.sigmoid(self.dec[1](h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        z_cond = torch.cat((z, x[:, -self.cond_data_len:]), dim=1)
        return self.decode(z_cond), mu, logvar


class VAE(nn.Module):
    def __init__(self, input_dim, bottleneck=5, inter_dim=40):
        nn.Module.__init__(self)

        self.fc1 = nn.Linear(input_dim, inter_dim)
        self.fc21 = nn.Linear(inter_dim, bottleneck)
        self.fc22 = nn.Linear(inter_dim, bottleneck)

        self.fc3 = nn.Linear(bottleneck, inter_dim)
        self.fc4 = nn.Linear(inter_dim, input_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

        self.input_dim = input_dim

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def to(self, device):
        self.device = device
        return super().to(device)

    def fit(self, batch_generator, epochs=1, max_iter=10e10, verbose=True):
        self.train()
        train_loss = []
        for epoch in range(epochs):
            with tqdm(batch_generator) as pbar:
                batch_idx = -1
                for batch in pbar:
                    batch_idx += 1
                    if batch_idx > max_iter:
                        print("Reached maximal number of iterations.")
                        break
                    batch = batch['image'].float()
                    data = T(batch).to(self.device)
                    self.optimizer.zero_grad()
                    recon_batch, mu, logvar = self(data)
                    loss = loss_function(recon_batch, data.view(-1, self.input_dim),
                                         mu, logvar, likelihood='bce')
                    loss.backward()
                    train_loss += [loss.item()/len(data)]
                    self.optimizer.step()
                    if verbose:
                        pbar.set_description(
                            f"Epoch {epoch}, Loss at batch {batch_idx:05d}: {loss.item()/len(data):.1f}"
                        )
        return train_loss


def loss_function(recon_x, x, mu, logvar, beta=1, likelihood='mse'):
    """
    Reconstruction + KL divergence losses summed over all elements
    and averaged over batch
    """
    if likelihood == 'bce':  ## Binary cross entropy
        rec_err = F.binary_cross_entropy(recon_x, x, reduction='sum')
    elif likelihood == 'mse': ## Mean squared error
        rec_err = torch.mean((recon_x - x)**2)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    KLD = torch.mean(KLD)
    return rec_err + beta * KLD


class convVAE(nn.Module):
    def __init__(self, bottleneck=7):
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
        self.to_mu = nn.Linear(80, bottleneck)
        self.to_sig = nn.Linear(bottleneck, 80)
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
        device='cpu', weight_fn=None, conditional=False):
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

    all_train_loss = []
    for epoch in range(epochs):
        for phase in ['train', 'val']:
            epoch_loss = []
            N = len(data_loader[phase].dataset)
            M = data_loader[phase].batch_size
            with ExitStack() as stack:
                pbar = stack.enter_context(tqdm(data_loader[phase]))
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                    stack.enter_context(torch.no_grad())

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
                    # loss = loss_function(recon_batch, data.view(-1, D+3),
                    #                      mu, logvar, beta=model.beta)
                    loss = loss_function(recon_batch, data,
                                         mu, logvar, beta=model.beta,
                                         likelihood=likelihood)
                    model.optimizer.zero_grad()
                    if phase == 'train':
                        loss.backward()
                        epoch_loss += [loss.item()]
                        model.optimizer.step()
                    else:
                        epoch_loss += [loss.item()]
                    if verbose:
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
