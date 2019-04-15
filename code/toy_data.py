import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class HierarchyImages(Dataset):
    """ Toy data for pose estimation """
    def __init__(self, angles=None,
                 bone_lengths=None,
                 key_marker_width=1,
                 img_shape=(28, 28)):
        """
        :param angles:
          List of angles for each bone in the hierarchy
          relative to its parent bone.
          Shape (batch_size N, bone_number D)
        :param bone_lengths:
          List of bonelengths.
        """
        self.angles = angles
        assert len(bone_lengths) == angles.shape[1]
        self.bone_lengths = bone_lengths

        self.include_origin = True
        self.img_shape = img_shape
        self.key_marker_width = key_marker_width

    def __len__(self):
        return len(self.angles)

    def __getitem__(self, idx):
        pose = self.angles[idx]
        coords = forward(pose, self.bone_lengths)
        img = keypoint_to_image(coords,
                                size=self.img_shape,
                                fwhm=self.key_marker_width,
                                include_origin=self.include_origin)
        sample = {'image': img, 'angles': pose}
        return sample

    def plot_image(self, idx):
        img = self.__getitem__(idx)['image']
        plt.imshow(img)
        plt.show()


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


def keypoint_to_image(coords, size=(28, 28), fwhm=2,
                      include_origin=False):
    """keypoint_to_image
    :param coords:
    :param fwhm:
      full-width-(at)-half-maximum
      FWHM = 2 \sqrt{2\ln 2} \sigma
    """
    h, w = size
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    img = np.zeros(size)
    t1 = np.arange(-h//2, h//2)
    t2 = np.arange(-w//2, w//2)
    X, Y = np.meshgrid(t1, t2)
    if not include_origin:
        coords = coords[1:]
    for x, y in coords:
        img = img + np.exp(-0.5/sigma**2*((x-X)**2 + (y-Y)**2))
    return img


def angle_batch_to_image(angle_batch, lengths, img_shape=(28, 28)):
    """angle_batch_to_image

    :param angle_batch:
        Shape (batch_size N, bone_number D)
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
    returns generator making batches of images based on angles.

    :param angles:
    :param lengths:
    :param batch_size:
    :param key_point_width:
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
