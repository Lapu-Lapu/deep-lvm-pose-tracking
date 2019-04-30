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

def mean_var_activation(x):
    N = x.shape[1]
    return x[:, :N//2], F.softplus(x[:, N//2:])

class cVAE(nn.Module):
    def __init__(self, input_dim, cond_data_len, latent_dim=5, hidden=40,
                 likelihood='bernoulli', pre_dim=200):
        nn.Module.__init__(self)
        self.likelihood = likelihood
        if likelihood in ['normal', 'cauchy', 'laplace']:
            dec_out_factor = 2
            out_activation = mean_var_activation
        else:
            dec_out_factor = 1
            out_activation = lambda x: (torch.sigmoid(x), None)

        # pre PCA
        self.pre = nn.ModulList([
            nn.Linear(input_dim, pre_dim)
            nn.Linear(pre_dim, input_dim))
        ])

        # Encoder
        self.enc = nn.ModuleList([
            nn.Linear(pre_dim, hidden),
            nn.Linear(hidden, 2 * latent_dim)
        ])
        self.enc_activations = [
            F.relu,
            # F.relu
            lambda x: x
        ]

        # Decoder
        self.dec = nn.ModuleList([
            nn.Linear(latent_dim + cond_data_len, hidden),
            nn.Linear(hidden, dec_out_factor * pre_dim)
            # nn.Linear(latent_dim + cond_data_len, dec_out_factor * input_dim)
        ])
        self.dec_activations = [
            F.relu,
            out_activation
        ]

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.cond_data_len = cond_data_len

    def prePCA(self, x):
        h = self.pre[0](x)
        rec = self.pre[1](h)
        return h, rec

    def encode(self, x):
        out = x
        for layer, activation in zip(self.enc, self.enc_activations):
            out = activation(layer(out))
        return out[:, :self.latent_dim], out[:, self.latent_dim:]

    def decode(self, z, observed):
        out = torch.cat((z, observed), dim=1)
        for layer, activation in zip(self.dec, self.dec_activations):
            out = activation(layer(out))
        return out

    def forward(self, x):
        h, rec = self.prePCA(x)
        mu, logvar = self.encode(h)
        z = reparameterize(mu, logvar)
        observed = x[:, -self.cond_data_len:]
        mu_obs, var_obs = self.decode(z, observed)
        return mu_obs, var_obs, mu, logvar, h, rec

    def loss(self, x, c, mu_obs, var_obs, mu, logvar, h, rec):
        prePCA = torch.mean((rec-x)**2)
        neg_llh_x = torch.mean((h-mu_obs)**2/var_obs)
        # TODO: add loss term for label.
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        KLD = torch.mean(KLD)
        return prePCA, neg_llh, KLD


class VAE(cVAE):
    def __init__(self, input_dim, latent_dim=5, hidden=40,
                 likelihood='bernoulli'):
        cVAE.__init__(self, input_dim, latent_dim=latent_dim,
                      hidden=hidden, cond_data_len=0,
                      likelihood=likelihood)

    def decode(self, z):
        out = z
        for layer, activation in zip(self.dec, self.dec_activations):
            out = activation(layer(out))
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        mu_x, var_x = self.decode(z)
        return mu_x, var_x, mu, logvar


def loss_function(decoded, x, mu, logvar, beta=1, likelihood='normal'):
    """
    Reconstruction + KL divergence losses summed over all elements
    and averaged over batch
    decoded: reconstructed input (batch_size, dimension)
    x: input (batch_size, dimension)
    mu: \mu(x) - mean of q(z|x, \phi) (batch_size, latent_dim)
    logvar: \log(\sigma^2) of latent variable of input (batch_size, latent_dim)
    """
    if likelihood == 'bernoulli':  ## Binary cross entropy
        # x_hat \log(x) + (1-x_hat) \log(1-x)
        rec_err = F.binary_cross_entropy(decoded[0], x, reduction='sum')
    elif likelihood == 'normal': ## Mean squared error
        mu_x, var_x = decoded
        rec_err = 0.5 * torch.mean((mu_x - x)**2/var_x)  # + 0.5*M*D*np.log(0.5*pi)
    elif likelihood == 'cauchy':
        x_0, gamma_sqr = decoded
        rec_err = -torch.log(gamma_sqr) + torch.log((x_0-x)**2+gamma_sqr)
        rec_err = torch.mean(rec_err)
    elif likelihood == 'laplace':
        mu_x, var_x = decoded
        rec_err = torch.mean(torch.abs(mu_x - x)/var_x + 2 * var_x)


    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # import pdb; pdb.set_trace()
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    KLD = torch.mean(KLD)
    return rec_err, beta * KLD


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
        loss_func=None, plotter=None):
    """
    model: instance of nn.Module
    data_loader: Dictionary of pytorch.util.data.DataSet for training and
                 validation
    """
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
                        label = batch['angles'].float().to(device)
                        # batch = torch.cat((img, label), dim=1).float()
                        # data = T(batch.float()).to(device)
                    else:
                        data = T(img.float()).to(device)

                    mu_x, var_x, mu, logvar, h, rec = model(data)
                    neg_ell, kl = loss_func(decoded=(mu_x, var_x), x=data, mu=mu, logvar=logvar)
                    prePCA, neg_ell, kl = model.loss(x, label, mu_x, var_x, mu, logvar, h, rec)

                    auxiliary_loss = False
                    if auxiliary_loss:
                        # https://www.reddit.com/r/MachineLearning/comments/al0lvl/d_variational_autoencoders_are_not_autoencoders/efaf4tl?utm_source=share&utm_medium=web2x
                        z = torch.normal(mean=torch.zeros((100, model.latent_dim)),
                                         std=torch.ones((100, model.latent_dim))).to('cuda')
                        x = model.decode(z)
                        # import pdb; pdb.set_trace()
                    aux = F.mse_loss(z, model.encode(x[0])[0]) if auxiliary_loss else 0

                    loss = neg_ell + kl + aux

                    optimizer.zero_grad()
                    if phase == 'train':
                        loss.backward()
                        prev_loss = loss.item()
                        if plotter is not None and batch_idx % 20 == 0:
                            plotter.plot('Loss', 'Val', 'Loss', len(epoch_loss), prev_loss)
                            plotter.plot('neg_ell', 'Val', 'neg_ell', len(epoch_loss), neg_ell.item())
                            plotter.plot('kl', 'Val', 'kl', len(epoch_loss), kl.item())
                            if auxiliary_loss:
                                plotter.plot('aux', 'Val', 'aux', len(epoch_loss), aux.item())
                            plotter.plot_image('reconstruction', mu_x)
                            plotter.plot_image('original', data)
                        epoch_loss += [prev_loss]
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
