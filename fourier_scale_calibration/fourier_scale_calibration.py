import numpy as np
from scipy import ndimage


# from fourier_scale_calibration.simulate import superpose_deltas


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
    f = np.fft.fft2(image * m)
    return f


def periodic_smooth_decomposition(I):
    u = I.astype(np.float64)
    v = u2v(u)
    v_fft = np.fft.fftn(v)

    s = v2s(v_fft)

    s_i = np.fft.ifftn(s)
    s_f = np.real(s_i)
    p = u - s_f
    return p, s_f


def u2v(u):
    v = np.zeros(u.shape, dtype=np.float64)

    v[0, :] = np.subtract(u[-1, :], u[0, :])
    v[-1, :] = np.subtract(u[0, :], u[-1, :])

    v[:, 0] += np.subtract(u[:, -1], u[:, 0])
    v[:, -1] += np.subtract(u[:, 0], u[:, -1])
    return v


def v2s(v_hat):
    M, N = v_hat.shape

    q = np.arange(M).reshape(M, 1).astype(v_hat.dtype)
    r = np.arange(N).reshape(1, N).astype(v_hat.dtype)

    den = (2 * np.cos(np.divide((2 * np.pi * q), M)) + 2 * np.cos(np.divide((2 * np.pi * r), N)) - 4)

    s = np.zeros_like(v_hat)
    s[den != 0] = v_hat[den != 0] / den[den != 0]
    s[0, 0] = 0
    return s


def detect_fourier_spots(image, template, symmetry, min_scale=None, max_scale=None, nbins_angular=None,
                         return_positions=False, normalize_radial=False, normalize_azimuthal=False,
                         ps_decomp=True):
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

    if nbins_angular is None:
        nbins_angular = int(np.ceil((2 * np.pi / symmetry) / 0.01))

    if ps_decomp:
       image, _ = periodic_smooth_decomposition(image)

    f = np.abs(np.fft.fft2(image))

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

    unrolled = (unrolled).mean(0)

    if normalize_azimuthal:
      unrolled = unrolled / unrolled.mean((1,), keepdims=True)

    if normalize_radial:
        unrolled = unrolled / unrolled.mean((0,), keepdims=True)

    p = np.unravel_index(np.argmax(unrolled), unrolled.shape)

    # import matplotlib.pyplot as plt
    # from scipy.ndimage import gaussian_filter
    # plt.figure()
    # plt.imshow(unrolled)
    # plt.show()
    # plt.figure()
    # plt.plot(unrolled.sum(1))
    # #plt.imshow(gaussian_filter(f,2))
    # #plt.plot(p[1],p[0],'ro')
    # plt.show()

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

    def __init__(self, template, lattice_constant, min_sampling=None, max_sampling=None,
                 normalize_radial=False, normalize_azimuthal=True):
        self.template = template
        self.lattice_constant = lattice_constant
        self.min_sampling = min_sampling
        self.max_sampling = max_sampling
        self.normalize_radial = normalize_radial
        self.normalize_azimuthal = normalize_azimuthal
        self._spots = None

    def get_spots(self):
        return self._spots

    def get_mask(self, shape, sigma):
        array = np.zeros(shape)
        spots = self.get_spots()[:, ::-1]
        superpose_deltas(spots, 0, array[None])
        array = np.fft.fftshift(array)
        x = np.fft.fftfreq(shape[0])
        y = np.fft.fftfreq(shape[1])
        z = np.exp(-(x[:, None] ** 2 + y[None] ** 2) * sigma ** 2 * 4)
        array = np.fft.ifft2(np.fft.fft2(array) * z).real
        return array

    def fourier_filter(self, image, sigma):
        spots = self.get_mask(image.shape, 4)
        spots /= spots.max()
        return np.fft.ifft2(np.fft.fft2(image) * spots).real

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
        elif self.template.lower() == 'ring':
            k = min(image.shape[-2:]) / self.lattice_constant * 2 / np.sqrt(3)
            sidelength = 2 * np.sin(np.pi / 128)
            template = regular_polygon(sidelength, 128)
            symmetry = 128
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

        scale, spots = detect_fourier_spots(image, template, symmetry, min_scale=min_scale, max_scale=max_scale,
                                            return_positions=True, normalize_azimuthal=self.normalize_azimuthal,
                                            normalize_radial=self.normalize_radial)

        self._spots = spots
        if return_spots:
            return scale / k, spots
        else:
            return scale / k

    def __call__(self, image):
        return self.calibrate(image)
