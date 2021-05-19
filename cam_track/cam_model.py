import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import cos, sin
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lmfit
import numba
import numpy as np
import scipy.ndimage as nd
import scipy.stats as st
from lmfit.model import ModelResult


@numba.jit
def gaussfit(x, y, x0, y0, A, sigma_x, sigma_y, theta, off):
    x = (x - x0)
    y = (y - y0)
    ct = cos(theta)
    st = sin(theta)
    sx = sigma_x * sigma_x
    sy = sigma_y * sigma_y
    a = ct**2 / (2 * sx) + st**2 / (2 * sy)
    b = sin(2 * theta) / 4 * (-1 / sx + 1 / sy)
    #b = -sin(2 * theta) / (4 * sx) + sin(2 * theta) / (4 * sy)
    c = st**2 / (2 * sx) + ct**2 / (2 * sy)
    expn = a * x**2 + 2 * b * x * y + c * y**2
    return A * np.exp(-expn) + off


gauss_mod = lmfit.Model(gaussfit, independent_vars=['x', 'y'])
for p in ['sigma_X', 'sigma_Y']:
    gauss_mod.set_param_hint(p, min=0)
gauss_mod.set_param_hint('theta', vary=0, min=0, max=np.pi / 4)


@dataclass
class Cam(ABC):
    name: str
    config: Dict[str, Any]
    last_image: Optional[np.ndarray] = None
    last_fit: Optional[ModelResult] = None

    def __post_init__(self):
        self.init_cam()

    def init_cam(self):
        pass

    @abstractmethod
    def read_cam(self) -> np.ndarray:
        raise NotImplementedError

    def get_com(self) -> Tuple[float, float]:
        if self.last_image is None:
            return np.nan, np.nan
        else:
            return nd.center_of_mass(self.last_image)

    def fit_gauss(self):
        x, y = np.indices(self.last_image.shape)
        x0, y0 = self.get_com()
        if self.last_fit is not None:
            d = self.last_fit.params
        else:
            d = {
                'A': self.last_image.max(),
                'sigma_x': 50,
                'sigma_y': 50,
                'theta': 0,
                'off': self.last_image.min(),
                'x0': x0,
                'y0': y0,
            }
        gauss_mod.set_param_hint('x0', min=0, max=self.last_image.shape[0])
        gauss_mod.set_param_hint('y0', min=0, max=self.last_image.shape[1])

        fr = gauss_mod.fit(self.last_image,
                           x=x,
                           y=y,
                           x0=d['x0'],
                           y0=d['y0'],
                           A=d['A'],
                           sigma_x=d['sigma_x'],
                           sigma_y=d['sigma_y'],
                           theta=d['theta'],
                           off=d['off'])
        self.last_fit = fr
        return fr


@dataclass
class MockCam(Cam):
    name: str = "TestCam"
    config: Dict[str, Any] = field(default_factory=dict)
    x_amp : float = 3
    y_amp : float = 3

    def read_cam(self) -> np.ndarray:
        t = dt.datetime.now()
        now = t.second + 1e-6 * t.microsecond        
        xc = self.x_amp * sin(now / 60.0 * 2 * np.pi * 3)
        yc = self.y_amp * cos(now / 60.0 * 2 * np.pi * 3)
        x, y = np.linspace(-10, 10, 320 // 2) - xc, np.linspace(
            -10 - yc, 10 + yc, 240 // 2) - yc
        gauss = np.exp(-0.5 * (x[:, None]**2 + y[None, :]**2) / (2**2))
        image = np.random.normal(loc=gauss * 120, scale=5) + 10
        self.last_image = image
        return image



