import sys
import time
import csv
import datetime
import numpy as np
import os
import pyqtgraph as pg 
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Slot, Qt

# --- CONFIGURATION ---
VITESSE = 60          # 60°/s
ANGLE_FLEX = 90       
ANGLE_EXT = 180       
FS = 500              
CHUNK = 20             

CALIB_FORCE_A = 147.68       
CALIB_ANGLE_A = 30.0  

try:
    from PyDAQmx import Task, DAQmx_Val_Diff, DAQmx_Val_Volts, DAQmx_Val_Rising, \
                        DAQmx_Val_ContSamps, DAQmx_Val_GroupByChannel, int32, byref
    HAS_DRIVER = True
except ImportError:
    HAS_DRIVER = False
    def int32(x=0): return x
    def byref(x): return x

class AcquisitionThread(QtCore.QThread):
    data_signal = QtCore.Signal(object, object)
    status_signal = QtCore.Signal(bool)

    def __init__(self):
        super().__init__()
        self.running = True
        self.is_rec = False
        self.is_simu = True
        self.current_event = "IDLE"
        self.csv_file = None
        self.csv_writer = None
        
        if HAS_DRIVER:
            try:
                self.task = Task()
                self.task.CreateAIVoltageChan("Dev8/ai3", "Force", DAQmx_Val_Diff, -10.0, 10.0, DAQmx_Val_Volts, None)
                self.task.CreateAIVoltageChan("Dev8/ai4", "Angle", DAQmx_Val_Diff, -10.0, 10.0, DAQmx_Val_Volts, None)
                self.task.CfgSampClkTiming("", FS, DAQmx_Val_Rising, DAQmx_Val_ContSamps, CHUNK)
                self.task.StartTask()
                self.is_simu = False
            except: self.is_simu = True

        self.read_buffer = np.zeros(CHUNK * 2, dtype=np.float64)
        self.sim_ang = 90.0
        self.sim_dir = 1

    def start_recording(self, filename):
        if not os.path.exists("DATA"): os.makedirs("DATA")
        self.csv_file = open(os.path.join("DATA", filename), 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Time', 'Force_N', 'Angle_Deg', 'Event'])
        self.is_rec = True

    def stop_recording(self):
        self.is_rec = False
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None

    def run(self):
        self.status_signal.emit(not self.is_simu)
        while self.running:
            if not self.is_simu:
                try:
                    read_count = int32()
                    self.task.ReadAnalogF64(CHUNK, 10.0, DAQmx_Val_GroupByChannel, self.read_buffer, CHUNK*2, byref(read_count), None)
                    force = self.read_buffer[:CHUNK] * CALIB_FORCE_A
                    angle = self.read_buffer[CHUNK:] * CALIB_ANGLE_A
                except: continue
            else:
                time.sleep(CHUNK/FS)
                step = (VITESSE/FS) * CHUNK
                self.sim_ang += step * self.sim_dir
                if self.sim_ang >= 180 or self.sim_ang <= 90: self.sim_dir *= -1
                angle = np.full(CHUNK, self.sim_ang)
                force = np.random.normal(60, 10, CHUNK) if self.sim_dir > 0 else np.random.normal(10, 2, CHUNK)

            self.data_signal.emit(force, angle)
            if self.is_rec and self.csv_writer:
                ts = time.time()
                for i in range(CHUNK):
                    self.csv_writer.writerow([ts, force[i], angle[i], self.current_event])

class BiodexMasterApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BIODEX MASTER - MVC CONCENTRIQUE")
        self.setMinimumSize(1100, 800)
        self.mvc_max = 0
        self.init_ui()
        
        self.thread = AcquisitionThread()
        self.thread.data_signal.connect(self.update_ui)
        self.thread.status_signal.connect(self.update_status)
        self.thread.start()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.countdown = 0

    def init_ui(self):
        self.setStyleSheet("QMainWindow { background-color: #121212; } QLabel { color: white; font-family: Segoe UI; }")
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # Contrôles
        panel = QtWidgets.QVBoxLayout()
        self.lbl_mode = QtWidgets.QLabel("RECHERCHE CARTE...")
        panel.addWidget(self.lbl_mode)

        box = QtWidgets.QGroupBox("COMMANDES")
        box.setStyleSheet("color: #3498db; font-weight: bold;")
        box_lay = QtWidgets.QVBoxLayout(box)
        
        self.btn_mvc = QtWidgets.QPushButton("LANCER MVC CONCENTRIQUE")
        self.btn_mvc.setStyleSheet("background-color: #e67e22; color: white; padding: 10px;")
        self.btn_mvc.clicked.connect(self.start_mvc)
        box_lay.addWidget(self.btn_mvc)

        self.btn_iso = QtWidgets.QPushButton("SÉRIE ISOCINÉTIQUE")
        self.btn_iso.setStyleSheet("background-color: #27ae60; color: white; padding: 10px;")
        self.btn_iso.clicked.connect(self.start_iso)
        box_lay.addWidget(self.btn_iso)
        panel.addWidget(box)

        self.lbl_timer = QtWidgets.QLabel("0.0 s")
        self.lbl_timer.setStyleSheet("font-size: 50px; color: #f1c40f;")
        panel.addWidget(self.lbl_timer)

        self.lbl_mvc = QtWidgets.QLabel("Peak Torque: 0 N")
        self.lbl_mvc.setStyleSheet("font-size: 20px; color: #e74c3c;")
        panel.addWidget(self.lbl_mvc)

        panel.addStretch()
        layout.addLayout(panel, 1)

        # Graphes
        vbox = QtWidgets.QVBoxLayout()
        self.pw_t = pg.PlotWidget(title="FORCE TEMPS RÉEL")
        self.curve_t = self.pw_t.plot(pen='g')
        vbox.addWidget(self.pw_t)

        
        self.pw_a = pg.PlotWidget(title="COURBE MOMENT / ANGLE")
        self.pw_a.setXRange(90, 180)
        self.curve_a = self.pw_a.plot(pen=pg.mkPen('#3498db', width=2))
        vbox.addWidget(self.pw_a)
        layout.addLayout(vbox, 3)

        self.ha, self.hf = [], []

    def update_status(self, hw):
        self.lbl_mode.setText("● MODE NI-DAQ" if hw else "● MODE SIMULATION")
        self.lbl_mode.setStyleSheet("color: #2ecc71;" if hw else "color: #e74c3c;")

    def start_mvc(self):
        self.mvc_max = 0
        self.run_proc("MVC_CONCENTRIC", 5)

    def start_iso(self):
        self.run_proc("ISOKINETIC_SERIE", 15)

    def run_proc(self, ev, dur):
        fn = f"{ev}_{datetime.datetime.now().strftime('%H%M%S')}.csv"
        self.thread.start_recording(fn)
        self.thread.current_event = ev
        self.countdown = dur
        self.timer.start(100)

    def tick(self):
        self.countdown -= 0.1
        if self.countdown <= 0:
            self.timer.stop()
            self.thread.stop_recording()
            self.thread.current_event = "IDLE"
        self.lbl_timer.setText(f"{max(0, self.countdown):.1f} s")

    @Slot(object, object)
    def update_ui(self, f, a):
        self.curve_t.setData(f)
        cur_f = np.mean(f)
        cur_a = a[-1]

        if self.thread.current_event == "MVC_CONCENTRIC" and 100 < cur_a < 170:
            if cur_f > self.mvc_max:
                self.mvc_max = cur_f
                self.lbl_mvc.setText(f"Peak Torque: {self.mvc_max:.1f} N")

        if 85 < cur_a < 185:
            self.ha.extend(a); self.hf.extend(f)
            if len(self.ha) > 1000: self.ha = self.ha[-1000:]; self.hf = self.hf[-1000:]
            self.curve_a.setData(self.ha, self.hf)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = BiodexMasterApp()
    win.show()
    sys.exit(app.exec())