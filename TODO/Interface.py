import sys
import time
import csv
import datetime
import numpy as np
import os
import pyqtgraph as pg 
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Slot, Qt

# --- CONFIGURATION MATÉRIELLE (D'après votre fichier de calibration) ---
FS = 500              
CHUNK = 20             
# Couple (Nm) = 147.7 * V + 0.5
CALIB_FORCE_A = 147.7       
CALIB_FORCE_B = 0.5          
# Angle : calculé sur la base de 0.3V=90° et 2.89V=0.5°
CALIB_ANGLE_A = -34.54       
CALIB_ANGLE_B = 100.36       

try:
    from PyDAQmx import Task, DAQmx_Val_Diff, DAQmx_Val_Volts, DAQmx_Val_Rising, \
                        DAQmx_Val_ContSamps, DAQmx_Val_GroupByChannel, int32, byref
    HAS_DRIVER = True
except ImportError:
    HAS_DRIVER = False
    def int32(x=0): return x
    def byref(x): return x

# --- THREAD D'ACQUISITION ---
class AcquisitionThread(QtCore.QThread):
    data_signal = QtCore.Signal(object, object)
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.is_simu = True
        self.task = None
        
        if HAS_DRIVER:
            try:
                self.task = Task()
                self.task.CreateAIVoltageChan("Dev1/ai3", "Force", DAQmx_Val_Diff, -10.0, 10.0, DAQmx_Val_Volts, None)
                self.task.CreateAIVoltageChan("Dev1/ai2", "Angle", DAQmx_Val_Diff, -10.0, 10.0, DAQmx_Val_Volts, None)
                self.task.CfgSampClkTiming("", FS, DAQmx_Val_Rising, DAQmx_Val_ContSamps, CHUNK)
                self.task.StartTask()
                self.is_simu = False
                print("DAQ Réelle connectée (Dev1)")
            except Exception as e:
                print(f"Mode Simulation activé (Erreur: {e})")
                self.is_simu = True

    def run(self):
        read_buffer = np.zeros(CHUNK * 2, dtype=np.float64)
        while self.running:
            if not self.is_simu:
                try:
                    read_count = int32()
                    self.task.ReadAnalogF64(CHUNK, 10.0, DAQmx_Val_GroupByChannel, read_buffer, CHUNK*2, byref(read_count), None)
                    force = (read_buffer[:CHUNK] * CALIB_FORCE_A) + CALIB_FORCE_B
                    angle = (read_buffer[CHUNK:] * CALIB_ANGLE_A) + CALIB_ANGLE_B
                except: continue
            else:
                time.sleep(CHUNK/FS)
                force = np.random.normal(20, 2, CHUNK)
                angle = np.full(CHUNK, 135.0)
            self.data_signal.emit(force, angle)

# --- INTERFACE PRINCIPALE ---
class BiodexInterface(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BIODEX CONTROL - PROTOCOLE RECHERCHE")
        self.setMinimumSize(1400, 850)
        
        # --- DATA & ÉTAT ---
        self.best_mvc = 0.0
        self.mvc_trials = []
        self.current_data = [] 
        self.is_recording = False
        self.current_rep = 0
        self.in_extension = False
        self.protocol_phase = 0
        self.force_offset = 0.0
        self.last_raw_force = 0.0
        self.current_trial_peak = 0.0

        # Buffer graphique (5 secondes)
        self.plot_max_points = 2500 
        self.all_force_data = np.zeros(self.plot_max_points)
        self.start_timestamp = time.time()
        
        self.init_ui()
        
        self.thread = AcquisitionThread()
        self.thread.data_signal.connect(self.update_plot)
        self.thread.start()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.countdown = 0
        self.max_time = 1 

    def init_ui(self):
        self.setStyleSheet("QMainWindow { background-color: #1a1a1a; } QLabel { color: white; } QPushButton { font-weight: bold; padding: 8px; }")
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # Panneau de contrôle
        side_panel = QtWidgets.QVBoxLayout()
        
        # Sujet
        sub_box = QtWidgets.QGroupBox("SUJET")
        sub_box.setStyleSheet("color: #3498db;")
        sub_lay = QtWidgets.QVBoxLayout(sub_box)
        self.name_input = QtWidgets.QLineEdit("SUJET_001")
        self.name_input.returnPressed.connect(self.validate_subject_name)
        sub_lay.addWidget(self.name_input)
        side_panel.addWidget(sub_box)

        # CMV
        mvc_box = QtWidgets.QGroupBox("MESURE CMV")
        mvc_box.setStyleSheet("color: #e67e22;")
        mvc_lay = QtWidgets.QVBoxLayout(mvc_box)
        self.btn_tare = QtWidgets.QPushButton("FAIRE LE ZÉRO (TARE)")
        self.btn_tare.clicked.connect(self.set_tare)
        mvc_lay.addWidget(self.btn_tare)
        self.btn_mvc = QtWidgets.QPushButton("LANCER CMV (3x5s)")
        self.btn_mvc.setStyleSheet("background-color: #d35400; color: white;")
        self.btn_mvc.clicked.connect(self.start_mvc_sequence)
        mvc_lay.addWidget(self.btn_mvc)
        self.lbl_mvc_res = QtWidgets.QLabel("MEILLEURE CMV : 0 N")
        mvc_lay.addWidget(self.lbl_mvc_res)
        side_panel.addWidget(mvc_box)

        # Protocole
        proto_box = QtWidgets.QGroupBox("PROTOCOLE")
        proto_box.setStyleSheet("color: #27ae60;")
        proto_lay = QtWidgets.QVBoxLayout(proto_box)
        self.btn_start_proto = QtWidgets.QPushButton("LANCER SÉRIE SUIVANTE")
        self.btn_start_proto.setStyleSheet("background-color: #27ae60; color: white;")
        self.btn_start_proto.clicked.connect(self.start_protocol_step)
        proto_lay.addWidget(self.btn_start_proto)
        side_panel.addWidget(proto_box)

        side_panel.addStretch()
        self.btn_csv = QtWidgets.QPushButton("EXPORTER EN CSV")
        self.btn_csv.clicked.connect(self.export_csv)
        self.btn_stop = QtWidgets.QPushButton("RÉINITIALISER")
        self.btn_stop.setStyleSheet("background-color: #c0392b; color: white;")
        self.btn_stop.clicked.connect(self.reset_all)
        side_panel.addWidget(self.btn_csv)
        side_panel.addWidget(self.btn_stop)
        main_layout.addLayout(side_panel, 1)

        # Zone Graphique
        right_panel = QtWidgets.QVBoxLayout()
        info_lay = QtWidgets.QHBoxLayout()
        self.lbl_status = QtWidgets.QLabel("IDLE")
        self.lbl_status.setStyleSheet("font-size: 20px; font-weight: bold; color: #f1c40f;")
        self.lbl_timer = QtWidgets.QLabel("0.0s")
        self.lbl_timer.setStyleSheet("font-size: 40px; color: white;")
        self.lbl_reps = QtWidgets.QLabel("REPS: 0/10")
        self.lbl_reps.setStyleSheet("font-size: 30px; color: #2ecc71;")
        info_lay.addWidget(self.lbl_status)
        info_lay.addWidget(self.lbl_timer)
        info_lay.addWidget(self.lbl_reps)
        right_panel.addLayout(info_lay)
        
        self.pbar_time = QtWidgets.QProgressBar()
        self.pbar_time.setFixedHeight(10)
        right_panel.addWidget(self.pbar_time)

        self.pw = pg.PlotWidget(title="FORCE (Nm)")
        self.pw.setBackground('#111')
        self.pw.setYRange(-20, 300)
        self.curve = self.pw.plot(pen=pg.mkPen('#2ecc71', width=2))
        self.target_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('r', width=2, style=Qt.DashLine))
        self.pw.addItem(self.target_line)
        right_panel.addWidget(self.pw)
        main_layout.addLayout(right_panel, 4)

    def validate_subject_name(self):
        self.lbl_status.setText(f"SUJET : {self.name_input.text()} VALIDÉ")

    def set_tare(self):
        self.force_offset = self.last_raw_force
        self.lbl_status.setText("ZÉRO EFFECTUÉ")

    def start_mvc_sequence(self):
        self.mvc_trial_count = 1
        self.mvc_trials = []
        self.btn_mvc.setEnabled(False)
        self.next_mvc_phase("PREPARATION (3s)", 3)

    def next_mvc_phase(self, status_text, seconds):
        self.lbl_status.setText(status_text)
        self.countdown = seconds
        self.max_time = seconds if seconds > 0 else 1
        self.timer.start(100)

    def tick(self):
        if self.countdown > 0:
            self.countdown -= 0.1
            self.pbar_time.setValue(int((self.countdown / self.max_time) * 100))
            self.lbl_timer.setText(f"{max(0, self.countdown):.1f}s")
        else:
            self.timer.stop()
            self.handle_phase_end()

    def handle_phase_end(self):
        status = self.lbl_status.text()
        if "PREPARATION" in status or "REPOS CMV" in status:
            self.current_trial_peak = 0
            self.is_recording = True
            self.next_mvc_phase(f"CMV ESSAi {self.mvc_trial_count}/3", 5)
        elif "CMV ESSAi" in status:
            self.is_recording = False
            self.mvc_trials.append(self.current_trial_peak)
            self.best_mvc = max(self.mvc_trials)
            self.lbl_mvc_res.setText(f"MEILLEURE CMV : {self.best_mvc:.1f} N")
            if self.mvc_trial_count < 3:
                self.mvc_trial_count += 1
                self.next_mvc_phase("REPOS CMV (10s)", 10)
            else:
                self.lbl_status.setText("CMV TERMINÉE")
                self.btn_mvc.setEnabled(True)
        elif "REPOS INTER-SÉRIE" in status:
            self.lbl_status.setText("PRÊT POUR SUITE")
            self.btn_start_proto.setEnabled(True)

    def start_protocol_step(self):
        if self.best_mvc <= 0:
            QtWidgets.QMessageBox.warning(self, "Erreur", "Faites une CMV d'abord.")
            return
        if self.protocol_phase == 0 or self.protocol_phase == 50: self.protocol_phase = 30
        else: self.protocol_phase = 50
        self.target_line.setValue(self.best_mvc * (self.protocol_phase / 100))
        self.lbl_status.setText(f"SERIE {self.protocol_phase}% EN COURS")
        self.current_rep = 0
        self.lbl_reps.setText("REPS: 0/10")
        self.is_recording = True
        self.btn_start_proto.setEnabled(False)

    def update_plot(self, force, angle):
        self.last_raw_force = np.mean(force)
        force_corrected = force - self.force_offset
        
        # Graphique défilant
        shift = len(force_corrected)
        self.all_force_data = np.roll(self.all_force_data, -shift)
        self.all_force_data[-shift:] = force_corrected
        self.curve.setData(self.all_force_data)
        
        cur_f = np.mean(force_corrected)
        cur_a = angle[-1]

        # Logique CMV
        if self.is_recording and "ESSAi" in self.lbl_status.text():
            if cur_f > self.current_trial_peak: self.current_trial_peak = cur_f

        # Détection Reps
        if cur_a > 55 and not self.in_extension:
            self.in_extension = True
            if self.is_recording and "SERIE" in self.lbl_status.text():
                self.current_rep += 1
                self.lbl_reps.setText(f"REPS: {self.current_rep}/10")
                if self.current_rep >= 10:
                    self.is_recording = False
                    dur = 180 if self.protocol_phase == 30 else 90
                    self.next_mvc_phase("REPOS INTER-SÉRIE", dur)
        elif cur_a < 120: self.in_extension = False

        if self.is_recording:
            ts = time.time() - self.start_timestamp
            self.current_data.append([ts, cur_f, cur_a, self.lbl_status.text()])

    def export_csv(self):
        fname = f"DATA_{self.name_input.text()}_{datetime.datetime.now().strftime('%H%M%S')}.csv"
        with open(fname, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Force_Nm', 'Angle_Deg', 'Phase'])
            writer.writerows(self.current_data)
        QtWidgets.QMessageBox.information(self, "Succès", f"Fichier {fname} créé.")

    def reset_all(self):
        self.timer.stop()
        self.best_mvc = 0
        self.protocol_phase = 0
        self.current_data = []
        self.lbl_status.setText("IDLE")
        self.lbl_reps.setText("REPS: 0/10")
        self.target_line.setValue(0)
        self.btn_mvc.setEnabled(True)
        self.btn_start_proto.setEnabled(True)

    def closeEvent(self, event):
        self.thread.running = False
        self.thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = BiodexInterface()
    win.show()
    sys.exit(app.exec())
