import numpy as np
from ase import Atoms
from typing import Sequence
from ase.build import graphene
from scipy.ndimage import gaussian_filter
from ase.lattice.hexagonal import *

def cut_rectangle(atoms: Atoms, origin: Sequence[float], extent: Sequence[float], margin: float = 0.):
    """
    Cuts out a cell starting at the origin to a given extent from a sufficiently repeated copy of atoms.

    Parameters
    ----------
    atoms : ASE atoms object
        This should correspond to a repeatable unit cell.
    origin : two float
        Origin of the new cell. Units of Angstrom.
    extent : two float
        xy-extent of the new cell. Units of Angstrom.
    margin : float
        Atoms within margin from the border of the new cell will be included. Units of Angstrom. Default is 0.

    Returns
    -------
    ASE atoms object
    """

    atoms = atoms.copy()
    cell = atoms.cell.copy()

    extent = (extent[0], extent[1], atoms.cell[2, 2],)
    atoms.positions[:, :2] -= np.array(origin)

    a = atoms.cell.scaled_positions(np.array((extent[0] + 2 * margin, 0, 0)))
    b = atoms.cell.scaled_positions(np.array((0, extent[1] + 2 * margin, 0)))

    repetitions = (int(np.ceil(abs(a[0])) + np.ceil(abs(b[0]))),
                   int(np.ceil(abs(a[1])) + np.ceil(abs(b[1]))), 1)

    shift = (-np.floor(min(a[0], 0)) - np.floor(min(b[0], 0)),
             -np.floor(min(a[1], 0)) - np.floor(min(b[1], 0)), 0)
    atoms.set_scaled_positions(atoms.get_scaled_positions() - shift)

    atoms *= repetitions

    atoms.positions[:, :2] -= margin

    atoms.set_cell([extent[0], extent[1], cell[2, 2]])

    atoms = atoms[((atoms.positions[:, 0] >= -margin) &
                   (atoms.positions[:, 1] >= -margin) &
                   (atoms.positions[:, 0] < extent[0] + margin) &
                   (atoms.positions[:, 1] < extent[1] + margin))
    ]
    return atoms


def make_amourphous_contamination(extent, filling, scale, density, margin=0, sampling=.05):
    shape = np.ceil(np.array((extent[0] + 2 * margin, extent[1] + 2 * margin)) / sampling).astype(np.int)
    sigma = np.max(np.array(shape) * scale) / 2

    noise = gaussian_filter(np.random.randn(*shape), sigma)

    hist, bin_edges = np.histogram(noise, bins=128)
    threshold = bin_edges[np.searchsorted(np.cumsum(hist), noise.size * (1 - filling))]

    contamination = np.zeros_like(noise)
    contamination[noise > threshold] += noise[noise > threshold] - threshold
    contamination = contamination / np.sum(contamination) * density * np.sum(contamination > 0.) * sampling ** 2

    positions = np.array(np.where(np.random.poisson(contamination) > 0.)).astype(np.float).T
    positions += np.random.randn(len(positions), 2)

    positions *= sampling
    positions -= margin
    positions = np.hstack((positions, np.zeros((len(positions), 1))))

    atoms = Atoms('C' * len(positions), positions=positions, cell=[extent[0], extent[1], 0])
    return atoms


def make_graphene(extent, margin=5, rotation=None, double_layer=False):

    if double_layer:
        atoms = Graphite(symbol='C', latticeconstant={'a': 2.46, 'c': 0}, size=(1, 1, 1))
    else:
        atoms = graphene()

    if rotation is None:
        rotation = np.random.rand() * 360

    atoms.rotate('z', rotation, rotate_cell=True)

    atoms = cut_rectangle(atoms, (0, 0), extent, margin=margin)
    return atoms
