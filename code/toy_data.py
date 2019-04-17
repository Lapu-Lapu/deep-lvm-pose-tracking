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
