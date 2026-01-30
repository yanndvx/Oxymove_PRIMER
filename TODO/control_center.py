import sys
import time
import csv
import datetime
import numpy as np
import os
import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration, LineGrouping
from nidaqmx.stream_readers import AnalogMultiChannelReader
import pyqtgraph as pg
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt, Slot
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop

# --- CONFIGURATION MATÉRIELLE ---
FS = 500
CHUNK = 20
NI_DEVICE_NAME = "Dev3"  # Ta carte pour l'occlusion
CHAN_ANALOG = f"{NI_DEVICE_NAME}/ai3, {NI_DEVICE_NAME}/ai0"
CHAN_VALVE_IN  = f"{NI_DEVICE_NAME}/port0/line0"
CHAN_VALVE_OUT = f"{NI_DEVICE_NAME}/port0/line1"

class GlobalAcquisition(QtCore.QThread):
    """Thread unique gérant le DAQ (Occlusion) et le LSL (Biodex)"""
    data_signal = QtCore.Signal(dict)

    def __init__(self):
        super().__init__()
        self.running = True
        self.inlet_force = None
        self.task = None
        
        # Initialisation DAQ (Dev3)
        try:
            self.task = nidaqmx.Task()
            self.task.ai_channels.add_ai_voltage_chan(
                CHAN_ANALOG, 
                terminal_config=TerminalConfiguration.RSE,
                min_val=-10.0, max_val=10.0
            )
            self.task.timing.cfg_samp_clk_timing(FS, sample_mode=AcquisitionType.CONTINUOUS)
            self.reader = AnalogMultiChannelReader(self.task.in_stream)
            self.task.start()
            self.data_buffer = np.zeros((2, CHUNK), dtype=np.float64)
        except Exception as e:
            print(f"Erreur DAQ : {e}")
            self.task = None

        # Ajoute ceci dans le __init__ du thread
        info_press = StreamInfo('OcclusionPressure', 'Pressure', 1, FS, 'float32', 'press_daq_001')
        self.outlet_press = StreamOutlet(info_press)

    def run(self):
        while self.running:
            payload = {'force_lsl': 0.0, 'pressure_daq': 0.0}

            # 1. Lecture DAQ (Pression)
            if self.task:
                try:
                    self.reader.read_many_sample(self.data_buffer, number_of_samples_per_channel=CHUNK)
                    raw_press = np.mean(self.data_buffer[1, :])
                    payload['pressure_daq'] = (raw_press * 0.625 - 0.625) * 750.1
                    self.outlet_press.push_sample([payload['pressure_daq']])
                except: pass
        
            # 2. Lecture LSL (Biodex Force)
            if self.inlet_force is None:
                streams = resolve_byprop('name', 'BiodexData', timeout=0.01)
                if streams: self.inlet_force = StreamInlet(streams[0])
            else:
                sample, ts = self.inlet_force.pull_sample(timeout=0.0)
                if sample: 
                    payload['force_lsl'] = sample[0] # On met juste la valeur dans le dictionnaire

            # On envoie le dictionnaire complet à l'interface
            self.data_signal.emit(payload)
            self.msleep(10)

    def stop(self):
        self.running = False
        if self.task: self.task.close()
        self.wait()

class ControlCenter(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BIODEX & OCCLUSION SYSTEM")
        self.setMinimumSize(1000, 800)
        self.setStyleSheet("QMainWindow { background-color: #1e1e1e; } QLabel { color: white; }")

        # Data
        self.mvc_val = 0.0
        self.plot_data = np.zeros(500)
        self.occ_duration = 5 * 60
        self.time_left = self.occ_duration

        # UI : Onglets
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        
        self.setup_force_tab()      # Ton script allégé
        self.setup_occlusion_tab()  # Ton nouveau script

        # Lancement Thread
        self.thread = GlobalAcquisition()
        self.thread.data_signal.connect(self.update_ui)
        self.thread.start()

        self.occ_timer = QtCore.QTimer()
        self.occ_timer.timeout.connect(self.tick_occlusion)

    def setup_force_tab(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(page)
        
        # Contrôles
        controls = QtWidgets.QVBoxLayout()
        self.input_mvc = QtWidgets.QDoubleSpinBox()
        self.input_mvc.setRange(0, 1000); self.input_mvc.setValue(100)
        btn_val_mvc = QtWidgets.QPushButton("VALIDER MVC")
        btn_val_mvc.clicked.connect(lambda: setattr(self, 'mvc_val', self.input_mvc.value()))
        
        controls.addWidget(QtWidgets.QLabel("VALEUR MVC (Nm)"))
        controls.addWidget(self.input_mvc)
        controls.addWidget(btn_val_mvc)
        
        for p in [0, 30, 50]:
            btn = QtWidgets.QPushButton(f"CIBLE {p}%")
            btn.clicked.connect(lambda chk, v=p: self.target_line.setValue((v/100)*self.mvc_val))
            controls.addWidget(btn)
        
        controls.addStretch()
        layout.addLayout(controls, 1)

        # Graph
        v_lay = QtWidgets.QVBoxLayout()
        self.lbl_force = QtWidgets.QLabel("0.0 Nm")
        self.lbl_force.setStyleSheet("font-size: 60px; font-weight: bold; color: #e74c3c;")
        self.pw = pg.PlotWidget()
        self.curve = self.pw.plot(pen=pg.mkPen('#e74c3c', width=3))
        self.target_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('#f1c40f', width=2, style=Qt.DashLine))
        self.pw.addItem(self.target_line)
        
        v_lay.addWidget(self.lbl_force, alignment=Qt.AlignCenter)
        v_lay.addWidget(self.pw)
        layout.addLayout(v_lay, 3)
        
        self.tabs.addTab(page, "MONITEUR FORCE")

    def setup_occlusion_tab(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        
        self.lbl_press = QtWidgets.QLabel("0.0 mmHg")
        self.lbl_press.setStyleSheet("font-size: 80px; color: #00e5ff; font-weight: bold;")
        
        self.lbl_timer = QtWidgets.QLabel("05:00")
        self.lbl_timer.setStyleSheet("font-size: 100px; color: white;")
        
        self.btn_start_occ = QtWidgets.QPushButton("DÉMARRER OCCLUSION")
        self.btn_start_occ.setStyleSheet("background-color: #27ae60; height: 80px; font-size: 20px;")
        self.btn_start_occ.clicked.connect(self.start_occlusion)

        self.btn_stop_occ = QtWidgets.QPushButton("ARRÊTER OCCLUSION")
        self.btn_stop_occ.setStyleSheet("""
            background-color: #c0392b; 
            height: 80px; 
            font-size: 20px;
            font-weight: bold;
        """)
        
        self.btn_stop_occ.clicked.connect(self.emergency_stop)
        self.btn_stop_occ.setEnabled(False)
        layout.addWidget(self.btn_stop_occ)
        layout.addWidget(self.lbl_press, alignment=Qt.AlignCenter)
        layout.addWidget(self.lbl_timer, alignment=Qt.AlignCenter)
        layout.addWidget(self.btn_start_occ)
        
        self.tabs.addTab(page, "CALIBRATION / OCCLUSION")

    def emergency_stop(self):
        self.occ_timer.stop()
        self.set_valves(False)
        
        self.lbl_timer.setText("STOP")
        self.lbl_timer.setStyleSheet("font_size: 100px; color : #ff1744;")
        self.btn_start_occ.setEnabled(True)
        self.btn_stop_occ.setEnabled(False)

        print ("ARRET D'URGENCE ACTIF")

                                     

    def start_occlusion(self):
        self.set_valves(True)
        self.time_left = self.occ_duration
        self.occ_timer.start(1000)
    
        self.lbl_timer.setStyleSheet("font-size: 100px; color: white;") # Reset couleur
        self.btn_start_occ.setEnabled(False)
        self.btn_stop_occ.setEnabled(True)
        

    def tick_occlusion(self):
        self.time_left -= 1
        mins, secs = divmod(self.time_left, 60)
        self.lbl_timer.setText(f"{mins:02d}:{secs:02d}")
        if self.time_left <= 0:
            self.occ_timer.stop()
            self.set_valves(False)
            self.btn_start_occ.setEnabled(True)

    def set_valves(self, active):
        try:
            with nidaqmx.Task() as t:
                t.do_channels.add_do_chan(CHAN_VALVE_IN)
                t.do_channels.add_do_chan(CHAN_VALVE_OUT)
                t.write([active, not active])
        except: print("Erreur Vannes")

    @Slot(dict)
    def update_ui(self, data):
        # 1. Récupération des valeurs du dictionnaire envoyé par le thread
        f = data['force_lsl']
        p = data['pressure_daq']
        
        # 2. Mise à jour des textes
        self.lbl_force.setText(f"{f:.1f} Nm")
        self.lbl_press.setText(f"{p:.1f} mmHg")
        
        # 3. Mise à jour du graphique (C'est ici que ça défile)
        self.plot_data = np.roll(self.plot_data, -1) # Décale tout vers la gauche
        self.plot_data[-1] = f                       # Ajoute la nouvelle valeur à droite
        self.curve.setData(self.plot_data)            # Rafraîchit le dessin

    def closeEvent(self, event):
        self.set_valves(False)
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = ControlCenter()
    win.show()
    sys.exit(app.exec())
