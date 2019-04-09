import numpy as np
import matplotlib.pyplot as plt


def forward(angles, armlengths=None):
    """forward
    Compute forward kinematics
    angles --> cartesian coordinates

    :param angles:
      List of angles for each bone in the hierarchy
      relative to its parent bone
    :param armlengths: List of bone lengths
    """
    if armlengths is None:
        armlengths = np.ones_like(angles)
    elif len(armlengths) == 2:
        armlengths = armlengths * np.ones_like(angles)
    else:
        try:
            assert len(angles) == len(armlengths)
        except AssertionError as e:
            raise Exception(f"Number of angles and armlengths should be the same"
                  f" but: {len(angles)} is not {len(armlengths)}")

    coords = [(0, 0)]
    cum_angle = 0
    for angle, bone in zip(angles, armlengths):
        offs = coords[-1]
        cum_angle += angle
        coords += [(bone * np.cos(cum_angle) + offs[0],
                    bone * np.sin(cum_angle) + offs[1])]
    return coords


def plot_keypoints(coords):
    """plot_keypoints

    :param coords:
    """
    for x, y in coords:
        plt.scatter(x, y)
    plt.legend(range(len(coords)))
    plt.grid()
    plt.ylim((-2, 2))
    plt.xlim((-2, 2))
    plt.show()


def keypoint_to_image(coords, size=(28, 28), beta=40,
                      include_origin=False):
    """keypoint_to_image
    """
    img = np.zeros(size)
    t1 = np.linspace(-2, 2, size[0])
    t2 = np.linspace(-2, 2, size[1])
    X, Y = np.meshgrid(t1, t2)
    if not include_origin:
        coords = coords[1:]
    for x, y in coords:
        img = img + np.exp(-beta*((x-X)**2 + (y-Y)**2))
    return img


def angle_batch_to_image(angle_batch, lengths, img_shape=(28, 28)):
    """angle_batch_to_image

    :param angle_batch:
    :param lengths:
    :rtype: np.array
    """
    h, w = img_shape
    img_batch = np.empty((len(angle_batch), 1, h, w))
    for i, angles in enumerate(angle_batch):
        coords = forward(angles, lengths)
        img_batch[i, 0] = keypoint_to_image(coords)
    return img_batch


def make_batch_generator(angles, lengths, batch_size, img_shape=(28, 28)):
    """make_batch_generator
    returns generator making batches of 28x28 images based
    on angles.

    :param angles:
    :param lengths:
    :param batch_size:
    """
    N = len(angles)
    for i in range(N//batch_size):
        angle_batch = angles[i*batch_size:(i+1)*batch_size]
        img_batch = angle_batch_to_image(angle_batch, lengths, img_shape)
        yield img_batch, angle_batch
    try:
        i += 1
    except UnboundLocalError:
        i = 0
    angle_batch = angles[i*batch_size:]
    if len(angle_batch) == 0 :
        raise StopIteration
    img_batch = angle_batch_to_image(angle_batch, lengths, img_shape)
    yield img_batch, angle_batch


def test_batch_generator():
    """test_batch_generator"""
    N = 10
    batch_size = 4
    labels = 2*np.pi*(np.random.rand(N, 2)-0.5)
    g = make_batch_generator(labels, [1, 2], N, batch_size)
    test = [4, 4, 2]
    out_shape = [len(batch_img) for batch_img, batch_pose in g]
    for t, o in zip(test, out_shape):
        assert t == o
