from tqdm import tqdm_notebook as tqdm

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch import Tensor as T

import numpy as np

from contextlib import ExitStack
from math import pi
from functools import partial


def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std

def mean_var_activation(x):
    """
    Activation function for distributions using 2 parameters,
    with the second one constraint to (0, \infty).
    E.g. Normal: mu and sigma
    """
    N = x.shape[1]
    return x[:, :N//2], F.softplus(x[:, N//2:])

class cVAE(nn.Module):
    def __init__(self, input_dim, pose_dim, latent_dim=5, hidden=40,
                 likelihood='bernoulli'):
        nn.Module.__init__(self)
        self.likelihood = likelihood
        if likelihood in ['normal', 'cauchy', 'laplace']:
            dec_out_factor = 2
            out_activation = mean_var_activation
        else:
            dec_out_factor = 1
            # use dummy output for consistency
            out_activation = lambda x: (torch.sigmoid(x), None)

        # Input
        self.inp_img = nn.Linear(input_dim, hidden//2)
        if pose_dim > 0:
            self.inp_pose = nn.Linear(pose_dim, hidden//2)

        # Encoder
        self.enc = nn.ModuleList([
            nn.Linear(hidden, 2 * latent_dim)
        ])
        self.enc_label = nn.ModuleList([
            nn.Linear(hidden, 2 * latent_dim)
        ])
        self.enc_activations = [
            lambda x: x
        ]

        # Decoder
        self.dec = nn.ModuleList([
            nn.Linear(latent_dim + pose_dim, hidden),
            nn.Linear(hidden, dec_out_factor * (input_dim+pose_dim))
            # nn.Linear(latent_dim + pose_dim, dec_out_factor * input_dim)
        ])
        self.dec_activations = [
            F.relu,
            out_activation
        ]

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.pose_dim = pose_dim

    def encode(self, x, p):
        out = F.relu(torch.cat((self.inp_img(x), self.inp_pose(p)), dim=1))
        for layer, activation in zip(self.enc, self.enc_activations):
            out = activation(layer(out))
        return out[:, :self.latent_dim], out[:, self.latent_dim:]

    def decode(self, z, observed):
        out = torch.cat((z, observed), dim=1)
        for layer, activation in zip(self.dec, self.dec_activations):
            out = activation(layer(out))
        return out

    def forward(self, x, pose):
        mu, logvar = self.encode(x, pose)
        z = reparameterize(mu, logvar)
        # observed = x[:, -self.pose_dim:]
        mu_obs, var_obs = self.decode(z, pose)
        mu_img = mu_obs[:, :self.input_dim]
        mu_label = mu_obs[:, self.input_dim:]
        if self.likelihood != 'bernoulli':
            var_img = var_obs[:, :self.input_dim]
            var_label = var_obs[:, self.input_dim:]
        else:
            var_img = None
            var_label = None
        return mu_img, var_img, mu_label, var_label, mu, logvar


class VAE(cVAE):
    def __init__(self, input_dim, latent_dim=5, hidden=40,
                 likelihood='bernoulli'):
        cVAE.__init__(self, input_dim, latent_dim=latent_dim,
                      hidden=hidden, pose_dim=0,
                      likelihood=likelihood)
        self.inp_img = nn.Linear(input_dim, hidden)

    def encode(self, x, p=None):
        out = F.relu(self.inp_img(x))
        for layer, activation in zip(self.enc, self.enc_activations):
            out = activation(layer(out))
        return out[:, :self.latent_dim], out[:, self.latent_dim:]

    def decode(self, z, observed=None):
        out = z
        for layer, activation in zip(self.dec, self.dec_activations):
            out = activation(layer(out))
        return out

    def forward(self, x, pose=None):
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
    img, pose  = x
    mu_img, var_img, mu_label, var_label = decoded
    if likelihood == 'bernoulli':  ## Binary cross entropy
        # x_hat \log(x) + (1-x_hat) \log(1-x)
        rec_err = F.binary_cross_entropy(mu_img, img, reduction='sum')
        rec_err += F.mse_loss(mu_label, pose, reduction='sum')
    elif likelihood == 'normal': ## Mean squared error
        rec_err = 0.5 * torch.mean((mu_img - img)**2/var_img)  # + 0.5*M*D*np.log(0.5*pi)
        rec_err += 0.5 * torch.mean((mu_label - pose)**2/var_label)
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
        loss_func = partial(loss_function, beta=1, likelihood=model.likelihood)

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

                    pose = T(batch['angles'].float()).to(device) if conditional else None
                    data = T(img.float()).to(device)

                    mu_img, var_img, mu_label, var_label, mu, logvar = model(data, pose)
                    neg_ell, kl = loss_func(decoded=(mu_img, var_img, mu_label, var_label),
                                            x=(data, pose), mu=mu, logvar=logvar)

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
                            plotter.plot_image('reconstruction', mu_img)
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
