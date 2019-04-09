from tqdm import tqdm

import torch
import torch.utils.data
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

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def to(self, device):
        self.device = device
        return super().to(device)

    def fit(self, batch_generator, epoch=1, max_iter=10e10):
        self.train()
        train_loss = []
        pbar = tqdm(enumerate(batch_generator))
        for batch_idx, (batch, _) in pbar:
            if batch_idx > max_iter:
                print("Reached maximal number of iterations.")
                break
            data = T(batch).to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += [loss.item()/len(data)]
            self.optimizer.step()
            pbar.set_description(
                f"Loss at batch {batch_idx:05d}: {loss.item()/len(data):.1f}"
            )
        return train_loss


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
