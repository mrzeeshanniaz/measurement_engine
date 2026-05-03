"""Minimal chumpy stub — makes SMPL PKL files loadable without the real chumpy package."""
import numpy as np

class Ch:
    """Stub for chumpy.Ch — stores the value as a plain numpy array."""
    def __init__(self, x=None, *args, **kwargs):
        self.x = np.array(x) if x is not None else np.array([])
        self._r = self.x

    @property
    def r(self):
        return self.x

    @property
    def shape(self):
        return self.x.shape

    @property
    def dtype(self):
        return self.x.dtype

    def __len__(self):
        return len(self.x)

    def __array__(self, dtype=None, copy=None):
        return self.x if dtype is None else self.x.astype(dtype)

    def __getitem__(self, idx):
        return self.x[idx]

    def __setitem__(self, idx, val):
        self.x[idx] = val

    def __repr__(self):
        return f"Ch({self.x!r})"

def depends_on(*args, **kwargs):
    return lambda f: f

reordering = type('module', (), {})()
combiners  = type('module', (), {})()
