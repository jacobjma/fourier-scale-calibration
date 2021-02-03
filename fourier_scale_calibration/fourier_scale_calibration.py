import numpy as np
from scipy import ndimage


def rotate(points, angle, center=None):
    if center is None:
        center = np.array([0., 0.])

    angle = angle / 180. * np.pi
    R = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = np.dot(R, points.T - np.array(center)[:, None]).T + center
    return points


def regular_polygon(sidelength, n):
    points = np.zeros((n, 2))

    i = np.arange(n, dtype=np.int32)
    A = sidelength / (2 * np.sin(np.pi / n))

    points[:, 0] = A * np.sin(i * 2 * np.pi / n)
    points[:, 1] = A * np.cos(-i * 2 * np.pi / n)

    return points


def cosine_window(x, cutoff, rolloff):
    rolloff *= cutoff
    array = .5 * (1 + np.cos(np.pi * (x - cutoff + rolloff) / rolloff))
    array[x > cutoff] = 0.
    array = np.where(x > cutoff - rolloff, array, np.ones_like(x))
    return array


def square_crop(image):
    shape = image.shape

    if image.shape[-1] != min(shape[-2:]):
        n = (image.shape[-2] - image.shape[-1]) // 2
        m = (image.shape[-2] - image.shape[-1]) - n
        image = image[..., n:-m, :]
    elif image.shape[-2] != min(shape[-2:]):
        n = (image.shape[-2] - image.shape[-1]) // 2
        m = (image.shape[-2] - image.shape[-1]) - n
        image = image[..., n:-m]

    return image


def windowed_fft(image):
    image = square_crop(image)

    x = np.fft.fftshift(np.fft.fftfreq(image.shape[-2]))
    y = np.fft.fftshift(np.fft.fftfreq(image.shape[-1]))
    r = np.sqrt(x[:, None] ** 2 + y[None] ** 2)
    m = cosine_window(r, .5, .33)
    return np.fft.fft2(image * m)


def detect_fourier_spots(image, template, symmetry, min_scale=None, max_scale=None, nbins_angular=128,
                         return_positions=False, normalize=True):
    if symmetry < 2:
        raise RuntimeError('symmetry must be 2 or greater')

    max_max_scale = (min(image.shape[-2:]) // 2) / np.max(np.linalg.norm(template, axis=1))

    if min_scale is None:
        min_scale = 1 / np.min(np.linalg.norm(template, axis=1))

    if max_scale is None:
        max_scale = max_max_scale

    else:
        max_scale = min(max_scale, max_max_scale)

    if min_scale > max_scale:
        raise RuntimeError('min_scale must be less than max_scale')

    f = np.abs(windowed_fft(image))
    if len(f.shape) == 3:
        f = f.mean(0)

    f = np.fft.fftshift(f)

    angles = np.linspace(0, 2 * np.pi / symmetry, nbins_angular, endpoint=False)
    scales = np.arange(min_scale, max_scale, 1)

    r = np.linalg.norm(template, axis=1)[:, None, None] * scales[None, :, None]
    a = np.arctan2(template[:, 1], template[:, 0])[:, None, None] + angles[None, None, :]

    templates = np.array([(np.cos(a) * r).ravel(), (np.sin(a) * r).ravel()])
    templates += np.array([f.shape[0] // 2, f.shape[1] // 2])[:, None]

    unrolled = ndimage.map_coordinates(f, templates, order=1)
    unrolled = unrolled.reshape((len(template), len(scales), len(angles)))

    if normalize:
        unrolled = unrolled / unrolled.mean((2,), keepdims=True)
    unrolled = (unrolled).mean(0)

    p = np.unravel_index(np.argmax(unrolled), unrolled.shape)

    if return_positions:
        r = np.linalg.norm(template, axis=1) * scales[p[0]]  # + min_scale
        a = np.arctan2(template[:, 1], template[:, 0])
        a -= p[1] * 2 * np.pi / symmetry / nbins_angular + np.pi / symmetry

        spots = np.array([(np.cos(a) * r).ravel(), (np.sin(a) * r).ravel()]).T
        spots += np.array([f.shape[0] // 2, f.shape[1] // 2])[None]
        return scales[p[0]], spots
    else:
        return scales[p[0]]


class FourierSpaceCalibrator:

    def __init__(self, template, lattice_constant, min_sampling=None, max_sampling=None, normalize=True):
        self.template = template
        self.lattice_constant = lattice_constant
        self.min_sampling = min_sampling
        self.max_sampling = max_sampling
        self.normalize = normalize

    def get_spots(self, image):
        return self.calibrate(image, return_spots=True)[1]

    def calibrate(self, image, return_spots=False):
        if self.template.lower() == 'hexagonal':
            k = min(image.shape[-2:]) / self.lattice_constant * 2 / np.sqrt(3)
            template = regular_polygon(1., 6)
            symmetry = 6
        elif self.template.lower() == '2nd-order-hexagonal':
            k = min(image.shape[-2:]) / self.lattice_constant * 2 / np.sqrt(3)
            template = regular_polygon(1., 6)
            template = np.vstack((template, rotate(template, 30) * np.sqrt(3)))
            symmetry = 6
        elif self.template.lower() == '12-sided':
            k = min(image.shape[-2:]) / self.lattice_constant * 2 / np.sqrt(3) / 2
            template = regular_polygon(1., 12)
            symmetry = 12
        else:
            raise NotImplementedError()

        if self.min_sampling is None:
            min_scale = None
        else:
            min_scale = k * self.min_sampling

        if self.max_sampling is None:
            max_scale = None
        else:
            max_scale = k * self.max_sampling

        if return_spots:
            scale, spots = detect_fourier_spots(image, template, symmetry, min_scale=min_scale, max_scale=max_scale,
                                                return_positions=return_spots, normalize=self.normalize)
            return scale / k, spots
        else:
            scale = detect_fourier_spots(image, template, symmetry, min_scale=min_scale, max_scale=max_scale,
                                         return_positions=False, normalize=self.normalize)
            return scale / k

    def __call__(self, image):
        return self.calibrate(image, return_spots=False)
