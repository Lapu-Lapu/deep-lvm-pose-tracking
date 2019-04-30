import matplotlib.pyplot as plt
import torch
import numpy as np
from visdom import Visdom

from itertools import product
from functools import partial

# useful for plotting on a 3x3 grid:
to_ind = np.array(list(product(range(3), range(3))))


def plot_reconstruction(recon, orig):
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(recon)
    ax[1].imshow(orig)
    ax[0].set_title('Reconstruction of an validation image.')
    ax[1].set_title('Original')
    plt.show()

def plot_sample_grid(sample, img_shape):
    assert len(sample) >= 9
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True)
    fig.set_size_inches((10, 10))
    for i in range(9):
        img = sample[i]
        ind = to_ind[i]
        ax[ind[0], ind[1]].imshow(img.reshape(*img_shape))
    plt.tight_layout()
    plt.show()

def draw_samples(model, observed=None):
    model.to('cpu')
    if observed is None:
        decode = model.decode
    else:
        observed = torch.Tensor(observed)
        decode = partial(model.decode, observed=observed)
    with torch.no_grad():
        sample = torch.randn(9, model.enc[-1].out_features//2)
        decoded = decode(sample)
        sample = decoded[0].numpy()
    return sample


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), 
                                                 Y=np.array([y,y]), 
                                                 env=self.env, 
                                                 opts=dict(
                                                     legend=[split_name],
                                                     title=title_name,
                                                     xlabel='Epochs',
                                                     ylabel=var_name)
                                                )
        else:
            self.viz.line(X=np.array([x]), 
                          Y=np.array([y]), 
                          env=self.env, 
                          win=self.plots[var_name], 
                          name=split_name, 
                          update = 'append')
            
    def plot_image(self, var_name, inp):
        inp = inp[0, :4096].cpu().detach().numpy()
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.heatmap(inp.reshape(64, 64))
        else:
            self.viz.heatmap(inp.reshape(64, 64),
                            env=self.env,
                            win=self.plots[var_name])
