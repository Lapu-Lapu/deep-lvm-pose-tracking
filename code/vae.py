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


def reparameterize(mu, logvar, anneal=1):
    std = torch.exp(0.5*logvar)
    eps = anneal*torch.randn_like(std)
    return mu + eps*std

def mean_var_activation(x):
    """
    Activation function for distributions using 2 parameters,
    with the second one constraint to (0, \infty).
    E.g. Normal: mu and sigma
    """
    N = x.shape[1]
    return F.elu(x[:, :N//2]), F.softplus(x[:, N//2:])

class cVAE(nn.Module):
    def __init__(self, input_dim, pose_dim, latent_dim=5, hidden=40,
                 likelihood='bernoulli', pre_dim=None):
        nn.Module.__init__(self)
        self.likelihood = likelihood
        if likelihood in ['normal', 'cauchy', 'laplace']:
            dec_out_factor = 2
            out_activation = mean_var_activation
        else:
            dec_out_factor = 1
            # use dummy output for consistency
            out_activation = lambda x: (torch.sigmoid(x), None)

        if pre_dim is None:
            pre_dim = input_dim
            self.prePCA = lambda x: (x, None)
            self.pre = 2*[lambda x: x]
        else:
            # pre PCA for image data
            def prePCA(x):
                h = self.pre[0](x)
                rec = self.pre[1](h)
                return h, rec
            self.prePCA = prePCA
            self.pre = nn.ModuleList([
                nn.Linear(input_dim, pre_dim),
                nn.Linear(pre_dim, input_dim)
            ])

        # Input
        self.inp_img = nn.Linear(pre_dim, hidden//2)
        if pose_dim > 0:
            self.inp_pose = nn.Linear(pose_dim, hidden//2)

        # Encoder
        self.enc = nn.ModuleList([
            nn.Linear(hidden, 2 * latent_dim)
        ])

        self.enc_activations = [
            # TODO: check encoder output activation in other implementations
            lambda x: x
        ]

        # Decoder
        self.dec = nn.ModuleList([
            nn.Linear(latent_dim + pose_dim, hidden),
            nn.Linear(hidden, dec_out_factor * (pre_dim+pose_dim))
            # nn.Linear(latent_dim + pose_dim, dec_out_factor * input_dim)
        ])
        self.dec_activations = [
            F.relu,
            out_activation
        ]

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.pre_dim = pre_dim
        self.pose_dim = pose_dim

    def encode(self, h, p):
        out = F.relu(torch.cat((self.inp_img(h), self.inp_pose(p)), dim=1))
        for layer, activation in zip(self.enc, self.enc_activations):
            out = activation(layer(out))
        return out[:, :self.latent_dim], out[:, self.latent_dim:]

    def decode(self, z, observed):
        out = torch.cat((z, observed), dim=1)
        for layer, activation in zip(self.dec, self.dec_activations):
            out = activation(layer(out))
        return out

    def _get_pose_param(self, mu_obs, var_obs):
        pose_param = {}
        pose_param['mean'] = mu_obs[:, self.pre_dim:]
        pose_param['var'] = (var_obs[:, self.pre_dim:]
                              if self.likelihood != 'bernoulli' else None)
        return pose_param

    def forward(self, x, pose=None, anneal=1):
        h, rec = self.prePCA(x)
        mu, logvar = self.encode(h, pose)
        z = reparameterize(mu, logvar, anneal=anneal)

        # shape (:, pre_dim+pose_dim), (:, pre_dim+pose_dim)
        mu_obs, var_obs = self.decode(z, pose)

        img_param, latent_param, pre_param = {}, {}, {}

        img_param['mean'] = mu_obs[:, :self.pre_dim]
        img_param['var'] = (var_obs[:, :self.pre_dim]
                            if self.likelihood != 'bernoulli' else None)

        pose_param = self._get_pose_param(mu_obs, var_obs)

        pre_param['latent'] = h
        pre_param['img'] = rec

        latent_param['mean'] = mu
        latent_param['logvar'] = logvar

        return img_param, latent_param, pose_param, pre_param


class VAE(cVAE):
    def __init__(self, input_dim, latent_dim=5, hidden=40,
                 pre_dim=None,
                 likelihood='bernoulli'):
        cVAE.__init__(self, input_dim, latent_dim=latent_dim,
                      hidden=hidden, pose_dim=0, pre_dim=pre_dim,
                      likelihood=likelihood)
        self.inp_img = nn.Linear(self.pre_dim, hidden)

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

    def _get_pose_param(self, mu_obs, var_obs):
        return None

    def forward_deprec(self, x, pose=None):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        mu_x, var_x = self.decode(z)
        return mu_x, var_x, mu, logvar


def joint_loss(img, pose, img_param, latent_param, pose_param, pre_param,
               likelihood='normal'):
    """
    Loss function including objective to learn PCA as preprocessing step
    TODO: decide what to do with covariances x_param['var'].
    """
    prePCA = (torch.mean((pre_param['img']-img)**2)
              if pre_param['img'] is not None else 0)

    if likelihood == 'bernoulli':
        neg_llh_img = F.binary_cross_entropy(img_param['mean'],
                                             pre_param['latent'], reduction='sum')
    elif likelihood == 'normal':
        neg_llh_img = torch.mean(
            (img_param['mean']-pre_param['latent'])**2  #/img_param['var']
        )
    else:
        raise(f'Observation model {likelihood} is not yet implemented')

    neg_llh_pose = torch.mean(
        (pose_param['mean']-pose)**2 #/ pose_param['var']
    ) if pose is not None else 0

    KLD = return_kl_term(latent_param)
    return prePCA, neg_llh_img, neg_llh_pose, KLD

def return_kl_term(latent_param):
    KLD = -0.5 * torch.sum(1 + latent_param['logvar']
                           - latent_param['mean'].pow(2)
                           - latent_param['logvar'].exp(), dim=1)
    KLD = torch.mean(KLD)
    return KLD


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

def pretrain(model, pre):
    for layer in model.pre:
        for param in layer.parameters():
            param.requires_grad = pre
    for layer in model.enc:
        for param in layer.parameters():
            param.requires_grad = not(pre)
    for layer in model.dec:
        for param in layer.parameters():
            param.requires_grad = not(pre)


def fit(model, data_loader, epochs=3, verbose=True, optimizer=None,
        device='cpu', weight_fn=None, conditional=False,
        loss_func=None, plotter=None, beta=1, stop_crit=1e-4):
    """
    model: instance of nn.Module
    data_loader: Dictionary of pytorch.util.data.DataSet for training and
                 validation
    """
    if optimizer is None:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    if loss_func is None:
        loss_func = partial(loss_function, beta=1, likelihood=model.likelihood)

    global pretrain
    pretrain = pretrain if model.pre_dim != model.input_dim else lambda x, y: 42

    stop = False  # is set to True if stopping criteria is fullfilled

    if model.pre_dim == model.input_dim:
        phases = ['train', 'val']
    else:
        phases = ['pretrain', 'train', 'val']

    all_train_loss = []
    anneal = 0.0001
    for epoch in range(epochs):
        if stop:
            break

        for phase in phases:
            if phase == 'pretrain' and epoch > 0:
                continue
            epoch_loss = []
            kls = []
            N = len(data_loader[phase].dataset)
            M = data_loader[phase].batch_size

            # conditional context manager
            with ExitStack() as stack:
                if phase in ['pretrain', 'train']:
                    model.train()
                    pbar = stack.enter_context(tqdm(data_loader[phase]))
                else:
                    model.eval()
                    stack.enter_context(torch.no_grad())
                    pbar = stack.enter_context(tqdm(data_loader[phase], disable=True))

                batch_idx = 0  # tqdm does not like enumerate
                for batch in pbar:
                    batch_idx += 1

                    # get data
                    img = batch['image'].view(data_loader[phase].batch_size, -1)
                    pose = T(batch['angles'].float()).to(device) if conditional else None
                    img = T(img.float()).to(device)

                    img_param, latent_param, pose_param, pre_param = model(img, pose)
                    prePCA, neg_llh_img, neg_llh_pose, kl = joint_loss(img, pose,
                                                                       img_param, latent_param,
                                                                       pose_param, pre_param)

                    if phase == 'train':
                        loss = neg_llh_img + neg_llh_pose + anneal * beta * kl
                        pretrain(model, False)
                    elif phase == 'val':
                        loss = neg_llh_img + neg_llh_pose + anneal * kl
                    else:
                        loss = prePCA
                        pretrain(model, True)

                    optimizer.zero_grad()
                    if phase in ['pretrain', 'train']:

                        with torch.autograd.detect_anomaly():
                            loss.backward()
                        prev_loss = loss.item()
                        if plotter is not None and batch_idx % 50 == 0:
                            visdom_plot(plotter, model, beta, epoch_loss,
                                        prev_loss, phase, prePCA, neg_llh_img,
                                        neg_llh_pose, kl, pre_param,
                                        img_param, img)

                            e_np = np.array(epoch_loss)
                            if (len(e_np) > 2000 and
                                np.abs(np.diff(e_np[-1900:]).mean()) < stop_crit):
                                print('Loss stopped decreasing.')
                                stop = True
                                break

                        epoch_loss += [prev_loss]
                        kls += [kl.item()]
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
    # return all_train_loss
    return epoch_loss*M/N  # last validation loss for gpyopt


def visdom_plot(plotter, model, beta, epoch_loss,
                prev_loss, phase, prePCA, neg_llh_img,
                neg_llh_pose, kl, pre_param,
                img_param, img):
    fmt = (beta, model.latent_dim)
    plotter.plot('Loss_{:.2f}_{}'.format(*fmt), 'Val', 'Loss',
                 len(epoch_loss), prev_loss)
    if phase != 'pretrain':
        if model.pre_dim != model.input_dim:
            plotter.plot('pca', 'Val', 'pca', len(epoch_loss), prePCA.item())

        plotter.plot(f'img_llh_{beta:.2f}_{model.latent_dim}',
                     'Val', 'img_llh',
                     len(epoch_loss), neg_llh_img.item())

        if model.pose_dim > 0:
            plotter.plot(f'pose_llh_{beta:.2f}_{model.latent_dim}',
                         'Val', f'pose_llh_{beta:.2f}_{model.latent_dim}',
                         len(epoch_loss), neg_llh_pose.item())
    else:
        plotter.plot_image('pca', pre_param['img'])

    plotter.plot('kl_{:.2f}_{}'.format(*fmt),
                 # 'Val', f'kl_{beta:.2f}_{model.latent_dim}',
                 'Val', 'kl',
                 len(epoch_loss), kl.item())

    plotter.plot_image('reconstruction', model.pre[1](img_param['mean']))
    plotter.plot_image('original', img)

if __name__ == '__main__':
    cVAE(1, 1)
