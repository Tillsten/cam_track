from pony.orm import Database, Required, Optional, Set
from datetime import datetime


def define_model():
    db = Database()


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
    
    return db#, Entry, CamLogEntry, FitResult