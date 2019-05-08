import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import numpy as np
from visdom import Visdom

from itertools import product
from functools import partial

from . import toy_data as toy

# useful for plotting on a 3x3 grid:
to_ind = np.array(list(product(range(3), range(3))))


class LatentTraverser(object):
    def __init__(self, model):
        self.model = model
        self.device = 'cpu'
        self.zeros = np.zeros((1, self.model.latent_dim))
        self.fig, self.ax = plt.subplots()
        self.d  = int(np.sqrt(self.model.input_dim))
        img = np.zeros((self.d, self.d))
        img[0, 0] = 1
        self.init_img = img
        self.mimg = plt.imshow(img)
        plt.close()

    def _init_img(self):
        self.mimg.set_data(self.init_img)
        return (self.mimg,)

    def _get_img(self, ind):
        zeros = np.zeros((1, self.model.latent_dim))
        for x in np.linspace(-4, 4):
            zeros[:, ind] = x
            mu_star = torch.Tensor(zeros).to(self.device)
            gen = self.model.decode(mu_star)[0]
            gen = gen.cpu().detach().numpy().reshape((self.d, self.d))
            # set_trace()
            yield (x, gen)

    def animate(self, img):
        x, img = img
        self.ax.set_title(f'{x:.3}')
        self.mimg.set_data(img)
        return (self.mimg,)

    def  get_anims(self, weight_dofs):
        anims = []
        for i in weight_dofs:
            generator = self._get_img(i)
            anim = FuncAnimation(self.fig, self.animate, init_func=self._init_img,
                                           frames=generator, interval=120,
                                           blit=True)
            anims += [anim]
        return anims


def make_bonelengths_and_width(n_bones=3, img_dim_sqrt=64):
    """
    Returns bone lengths and keymarker width with sensible defaults.
    """
    eps = np.random.rand(n_bones)
    bone_lengths = img_dim_sqrt//6 * (eps/2+1-1/n_bones)
    key_marker_width = 1.5 * img_dim_sqrt/32
    return bone_lengths, key_marker_width


def make_poses(N=36000):
    """
    Returns poses restricted such that there is no ambiguity.
    """
    poses = 1/2*np.pi*(np.random.rand(N, 3)-0.5)
    poses[:, 0] = poses[:, 0] * 4
    return poses


def pose_to_image(pose, bone_lengths, d):
    img = toy.keypoint_to_image(
        toy.forward(2*np.pi*(pose-0.5) , bone_lengths),
        size=(d, d),
        include_origin=True
    )
    return img

def un_normalize(poses):
    return 360*(poses-0.5)

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

def compose(func1, func2):
    return lambda x: func1(func2(x))

def return_first(f):
    return lambda y: (f(y)[0])

def draw_samples(model, observed=None):
    model.to('cpu')
    if observed is None:
        decode = model.decode
    else:
        observed = torch.Tensor(observed)
        decode = partial(model.decode, observed=observed)

    decode = return_first(decode)
    if model.pre_dim > 0:
        decode = compose(model.pre[1], decode)
        # decode = compose(decode, lambda x: x)

    with torch.no_grad():
        sample = torch.randn(9, model.enc[-1].out_features//2)
        decoded = decode(sample)
        sample = decoded.numpy()
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
