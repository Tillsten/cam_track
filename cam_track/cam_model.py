from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import scipy.ndimage as nd
import scipy.stats as st
import lmfit
from lmfit.model import ModelResult
from math import cos, sin

import numba
import datetime as dt


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
gauss_mod.set_param_hint('theta', min=0, max=np.pi / 4)


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

    def read_cam(self) -> np.ndarray:
        t = dt.datetime.now()
        now = t.second+1e-6*t.microsecond
        print(now)
        xc = 3*sin(now/60.0*2*np.pi*3)
        yc = 3*cos(now/60.0*2*np.pi*3)
        x, y = np.linspace(-10, 10, 320//2)-xc, np.linspace(-10-yc, 10+yc, 240//2)-yc
        gauss = np.exp(-0.5 * (x[:, None]**2 + y[None, :]**2) / (2**2))
        image = np.random.normal(loc=gauss * 120, scale=5) + 10
        self.last_image = image
        return image


from pony.orm import Database, Required, Optional, db_session, set_sql_debug, select, Set
from datetime import datetime

db = Database()
#set_sql_debug(True)


class Entry(db.Entity):
    date_added = Required(datetime)
    cam_entries = Set('CamLogEntry')


class CamLogEntry(db.Entity):
    entry = Required(Entry)
    cam = Required(str)
    loc_x = Required(float)
    loc_y = Required(float)
    mean = Required(float)
    max = Required(float)
    fit_result = Optional(lambda: FitResult)


class FitResult(db.Entity):
    cam_entry = Required(CamLogEntry)
    x0 = Required(float)
    y0 = Required(float)
    sigma_x = Required(float)
    sigma_y = Required(float)
    theta = Required(float)
    A = Required(float)
    off = Required(float)

from pathlib import Path 

if (p := Path('test.db')).exists():
    p.unlink()



@dataclass
class Tracker:
    #db: Database
    fname: str
    cams: List[Cam]
    interval_s: float = 1

    def __post_init__(self):
        self.open_db()

    def open_db(self, fname=None):                
        db.disconnect()
        if fname is not None:
            self.fname = fname
        db.bind('sqlite', self.fname, create_db=True)
        db.generate_mapping(create_tables=True)

    @db_session
    def track(self):
        for c in self.cams:
            c.read_cam()
            com = c.get_com()

            entry = Entry(date_added=datetime.now(), cam_entries=[])

            centry = CamLogEntry(
                entry=entry,
                cam=c.name,
                loc_x=com[0],
                loc_y=com[1],
                mean=c.last_image.mean(),
                max=c.last_image.max(),
            )
            centry.entry = entry

            fr = c.fit_gauss()
            fre = FitResult(cam_entry=centry,
                      x0=fr.params['x0'],
                      y0=fr.params['y0'],
                      sigma_x=fr.params['sigma_X'],
                      sigma_y=fr.params['sigma_Y'],
                      A=fr.params['A'],
                      theta=fr.params['theta'],
                      off=fr.params['off'])
            centry.fit_result = fre

    @db_session
    def get_param_history(self, cam: str, pname: str):        
        q = select((e.entry.date_added, getattr(e.fit_result, pname)) for e in CamLogEntry)
        x, y = [], []
        for (d, v) in q:
            #x.append(np.datetime64(d))
            x.append(d.timestamp())
            y.append(v)
        return np.array(x), np.array(y)

        


if __name__ == '__main__':
    tc = MockCam()
    tc.read_cam()
    tracker = Tracker(cams=[tc], fname='bla.db')
    for i in range(10):
        tracker.track()
    tracker.get_param_history('TestCam', 'x0')
    #print(tc.last_fit.fit_report())
    print(tracker.get_param_history('TestCam', 'x0'))
    #with db_session:
    #print(CamLogEntry[0])
