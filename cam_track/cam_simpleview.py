from typing import Optional, Tuple, List
import numpy as np
from pyqtgraph import (ROI, DateAxisItem, EllipseROI, ImageItem, IsocurveItem,
                       PlotWidget, colormap, makeARGB)
from pyqtgraph.graphicsItems.TargetItem import TargetItem
from qtpy.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (QAction, QApplication, QHBoxLayout, QMainWindow,
                            QPushButton, QStyle, QToolBar, QVBoxLayout,
                            QFileDialog, QWidget)
from pathlib import Path
from cam_track.cam_model import Cam
from cam_track.tracker_model import Tracker


class CamView(PlotWidget):
    def __init__(self, cam: Cam, parent=None) -> None:
        super().__init__(parent=parent)
        self.cam = cam
        self.main_plot_item = self.plotItem
        self.last_centers: List[Tuple[float]] = []
        self.image_item = ImageItem(cam.read_cam())
        cmap = colormap.get('CET-C2')
        #self.image_item.setLookupTable(cmap.getLookupTable())

        self.target = TargetItem(movable=False)
        self.main_plot_item.addItem(self.image_item)
        self.main_plot_item.addItem(self.target)

        self.iso_curve = IsocurveItem()
        self.iso_curve.setParentItem(self.image_item)
        self.main_plot_item.addItem(self.iso_curve)
        self.center_hist = self.main_plot_item.plot(pen='r')

        self.update_cam()

    @Slot()
    def update_cam(self):
        #print('update')
        image = self.cam.last_image
        self.image_item.setImage(image)
        if not self.cam.last_fit is None:
            lf = self.cam.last_fit
            center = (lf.params['x0'], lf.params['y0'])
            self.target.setPos(center)
            self.last_centers = [center] + self.last_centers[:30]
            self.iso_curve.setData(lf.best_fit, lf.best_fit.max() * 0.5)
            arr = np.array(self.last_centers)
            self.center_hist.setData(x=arr[:, 0], y=arr[:, 1])

    @Slot()
    def reset_hist(self):
        self.last_centers = []


class HistView(QWidget):
    def __init__(self, tracker: Tracker, parent=None):
        super().__init__(parent=parent)
        self.tracker = tracker
        params = ['x0', 'y0', 'sigma_x', 'sigma_y', 'A', 'theta']
        self.buttons = {}
        self.cur_param: Optional[str] = None
        hlay = QHBoxLayout(self)
        layout = QVBoxLayout()
        hlay.addLayout(layout)
        self.setLayout(hlay)
        hlay.setContentsMargins(0, 0, 0, 0)
        for l in params:
            self.buttons[l] = QPushButton(l)
            self.buttons[l].clicked.connect(lambda x, l=l: self.show_param(l))
            layout.addWidget(self.buttons[l])
        layout.addStretch(1)
        self.plot_widget = PlotWidget(self)
        ax = DateAxisItem()
        self.plot_widget.plotItem.setAxisItems({'bottom': ax})
        hlay.addWidget(self.plot_widget)

    def show_param(self, s: str):
        self.cur_param = s

    @Slot()
    def update_plot(self):
        if self.cur_param is None:
            return
        self.plot_widget.plotItem.clear()
        for c in self.tracker.cams:
            x, y = self.tracker.get_param_history('c', self.cur_param)
        self.plot_widget.plot(x, y)


class Controls(QWidget):
    def __init__(self, tracker, parent=None) -> None:
        super().__init__(parent=parent)
        self.tracker = tracker
        hl = QHBoxLayout()
        vl = QVBoxLayout()
        self.load_db = QPushButton('New file')
        self.export_db = QPushButton('Export as text')

        hl.addLayout(vl)

        self.setLayout(hl)
        hv = HistView(tracker.tracker, self)
        hl.addWidget(hv)
        self.tracker.new_entry.connect(hv.update_plot)

        for c in tracker.tracker.cams:
            cv = CamView(cam=c, parent=self)
            self.tracker.new_entry.connect(cv.update_cam)
            hl.addWidget(cv)


class QtTracker(QObject):
    tracking_started = Signal()
    tracking_stopped = Signal()
    new_entry = Signal()
    rest = Signal()

    def __init__(self, tracker: Tracker, parent=None):
        QObject.__init__(self, parent)
        self.tracker = tracker

    @Slot()
    def start_recording(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.track)
        self.timer.start(300)
        self.tracking_started.emit()

    def track(self):
        self.tracker.track()
        self.new_entry.emit()

    @Slot()
    def stop_recording(self):
        self.timer.stop()
        self.tracking_stopped.emit()


class Main(QMainWindow):
    def __init__(self, tracker: Tracker) -> None:
        super().__init__()
        self.qttracker = QtTracker(tracker)
        self.bg_thread = QThread(self)
        self.qttracker.moveToThread(self.bg_thread)
        self.bg_thread.start()
        self.qttracker.start_recording()

        self.setCentralWidget(Controls(self.qttracker))
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.new_action = QAction("New", self)
        self.new_action.setToolTip("Create a new file for tracking.")
        icon = self.style().standardIcon(QStyle.SP_FileIcon)
        self.new_action.setIcon(icon)
        self.new_action.triggered.connect(self.new_db)
        toolbar.addAction(self.new_action)

        self.load_action = QAction("Load", self)
        self.load_action.setToolTip("Load a existing file for tracking.")
        icon = self.style().standardIcon(QStyle.SP_DialogOpenButton)
        self.load_action.setIcon(icon)
        self.load_action.triggered.connect(self.load_db)
        toolbar.addAction(self.load_action)

    def load_db(self):
        fname, ftype = QFileDialog.getOpenFileName(
            self,
            "Tracking files",
            str(Path.home()),
            "CamTrack Files (*.camtrack)",
            options=QFileDialog.DontUseNativeDialog)
        if fname != "":
            self.qttracker.tracker.open_db(fname)


    def new_db(self):
        fname, ftype = QFileDialog.getSaveFileName(
            self, "New file",     
            str(Path.home()),
            "CamTrack Files (*.camtrack)",
            options=QFileDialog.DontUseNativeDialog)
        self.qttracker.tracker.open_db(fname)
        


if __name__ == '__main__':
    import qtmodern.styles
    import qtmodern.windows

    from cam_track.cam_model import MockCam

    app = QApplication([])
    qtmodern.styles.dark(app)

    cams = [MockCam()]
    tracker = Tracker(str(Path.home() / 'test.camtrack'), cams)

    thr = QThread()
    m = Main(tracker)
    #mw = qtmodern.windows.ModernWindow(m)
    m.show()
    app.exec_()
