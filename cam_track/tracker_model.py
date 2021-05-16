from dataclasses import dataclass
import datetime as dt
from pony.orm import db_session, select, set_sql_debug
from cam_track.db_model import CamLogEntry, Entry, FitResult, db
from typing import List, Iterable
from cam_track.cam_model import Cam

@dataclass
class Tracker:
    #db: Database
    fname: str
    cams: Iterable[Cam]
    interval_s: float = 1

    def __post_init__(self):
        self.open_db()

    def open_db(self, fname=None):
        db.()
        if fname is not None:
            self.fname = fname
        db.bind('sqlite', self.fname, create_db=True)
        db.generate_mapping(create_tables=True)

    @db_session
    def track(self):
        for c in self.cams:
            c.read_cam()
            com = c.get_com()

            entry = Entry(date_added=dt.datetime.now(), cam_entries=[])

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
        q = select((e.entry.date_added, getattr(e.fit_result, pname))
                   for e in CamLogEntry)
        x, y = [], []
        for (d, v) in q:
            #x.append(np.datetime64(d))
            x.append(d.timestamp())
            y.append(v)
        return x, y


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
