from dataclasses import dataclass, field
import datetime as dt
from pathlib import Path
from pony.orm import db_session, select, set_sql_debug, left_join

from cam_track.db_model import define_model  #CamLogEntry, Entry, FitResult, db
from typing import Any, List, Iterable
from cam_track.cam_model import Cam


@dataclass
class Tracker:

    fname: str
    cams: Iterable[Cam]
    interval_s: float = 1
    db: Any = field(default_factory=define_model)

    def __post_init__(self):
        self.open_db()

    def open_db(self, fname=None):
        self.db = define_model()
        if fname is not None:
            self.fname = fname
        self.db.bind('sqlite', self.fname, create_db=True)
        self.db.generate_mapping(create_tables=True)

    @db_session
    def track(self):
        db = self.db
        for c in self.cams:
            c.read_cam()
            com = c.get_com()

            entry = db.Entry(date_added=dt.datetime.now(), cam_entries=[])

            centry = db.CamLogEntry(
                entry=entry,
                cam=c.name,
                loc_x=com[0],
                loc_y=com[1],
                mean=c.last_image.mean(),
                max=c.last_image.max(),
            )
            centry.entry = entry

            fr = c.fit_gauss()
            fre = db.FitResult(cam_entry=centry,
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
        q = select((e.entry.date_added, getattr(e.fit_result, pname))
                   for e in self.db.CamLogEntry if e.cam == cam)
        x, y = [], []
        for (d, v) in q:
            #x.append(np.datetime64(d))
            x.append(d.timestamp())
            y.append(v)
        return x, y

    @db_session
    def text_export(self, fname: str):
        q = select((fr.cam_entry.entry.date_added, fr.cam_entry.cam, fr)
                   for fr in self.db.FitResult)
        n = len(self.db.FitResult[1].to_dict())
        d = self.db.FitResult[1].to_dict()
        with Path(fname).open('w') as f:
            f.write("Datetime, Cam")
            for k in d:
                f.write(", %s" % k)
            f.write('\n')
            for (date, cam, fr) in q:
                f.write(f"{date.isoformat(timespec='milliseconds')}, {cam}, ")
                for i, v in enumerate(fr.to_dict().values()):
                    if i < n - 1:
                        f.write('%.1f, ' % v)
                    else:
                        f.write('%.1f\n' % v)

    @db_session
    def pandas_export(self):
        import pandas as pd

if __name__ == '__main__':
    from cam_track.cam_model import MockCam
    tc = MockCam()
    tc.read_cam()
    tracker = Tracker(cams=[tc], fname='bla.db')
    for i in range(10):
        tracker.track()
    tracker.get_param_history('TestCam', 'x0')
    #print(tc.last_fit.fit_report())
    #print(tracker.get_param_history('TestCam', 'x0'))
    tracker.text_export('bla')
    #with db_session:
    #print(CamLogEntry[0])
