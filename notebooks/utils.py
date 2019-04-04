import numpy as np
import matplotlib.pyplot as plt
import PIL
from tqdm import tqdm

import torch
import torch.utils.data
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from torch import Tensor as T


class VAE(nn.Module):
    def __init__(self, bottleneck=5):
        nn.Module.__init__(self)

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, bottleneck)
        self.fc22 = nn.Linear(400, bottleneck)
        self.fc3 = nn.Linear(bottleneck, 400)
        self.fc4 = nn.Linear(400, 784)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
        pass

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def to(self, device):
        self.device = device
        return super().to(device)

    def fit(self, batch_generator, epoch=1):
        self.train()
        train_loss = 0
        pbar = tqdm(enumerate(batch_generator))
        for batch_idx, (batch, _) in pbar:
            data = T(batch).to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            pbar.set_description(
                f"Loss at batch {batch_idx:05d}: {loss.item()/len(data):.1f}"
            )
        return train_loss


def forward(angles, armlengths=None):
    """
    Compute forward kinematics
    angles --> cartesian coordinates
    """
    if armlengths is None:
        armlengths = np.ones_like(angles)
    else:
        assert len(angles) == len(armlengths)
    coords = [(0, 0)]
    for angle, bone in zip(angles, armlengths):
        offs = coords[-1]
        coords += [(bone * np.cos(angle) + offs[0],
                               bone *  np.sin(angle) + offs[1])]
    return coords


def plot_keypoints(coords):
    for i, (x, y) in enumerate(coords):
        plt.scatter(x, y)
    plt.legend(range(len(coords)))
    plt.grid()
    plt.ylim((-2, 2))
    plt.xlim((-2, 2))
    plt.show()


def keypoint_to_image(coords, size=(28, 28), beta=40,
                      include_origin=False):
    img = np.zeros(size)
    t1 = np.linspace(-2, 2, size[0])
    t2 = np.linspace(-2, 2, size[1])
    X, Y = np.meshgrid(t1, t2)
    if not include_origin:
        coords = coords[1:]
    for x, y in coords:
        img = img + np.exp(-beta*((x-X)**2+(y-Y)**2))
    return img


def angle_batch_to_image(angle_batch, lengths):
    img_batch = np.empty((len(angle_batch), 1, 28, 28))
    for i, angles in enumerate(angle_batch):
        alpha, beta = angles
        coords = forward(angles, lengths)
        img_batch[i, 0] = keypoint_to_image(coords)
    return img_batch


def make_batch_generator(angles, lengths, N, batch_size):
    for i in range(N//batch_size):
        angle_batch = angles[i*batch_size:(i+1)*batch_size]
        img_batch = angle_batch_to_image(angle_batch, lengths)
        yield img_batch, angle_batch
    i += 1
    angle_batch = angles[i*batch_size::]
    img_batch = angle_batch_to_image(angle_batch, lengths)
    yield img_batch, angle_batch


def test_batch_generator():
    N = 10
    batch_size = 4
    labels = 2*np.pi*(np.random.rand(N, 2)-0.5)
    g = batch_generator(labels, N, batch_size)
    test = [4, 4, 2]
    out_shape = [len(batch_img) for batch_img, batch_pose in g]
    for t, o in zip(test, out_shape):
        assert t == o


def loss_function(recon_x, x, mu, logvar):
    """
    Reconstruction + KL divergence losses summed over all elements and batch
    """
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
