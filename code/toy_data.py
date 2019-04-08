import numpy as np
import matplotlib.pyplot as plt


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
                    bone * np.sin(angle) + offs[1])]
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
    g = make_batch_generator(labels, [1, 2], N, batch_size)
    test = [4, 4, 2]
    out_shape = [len(batch_img) for batch_img, batch_pose in g]
    for t, o in zip(test, out_shape):
        assert t == o
