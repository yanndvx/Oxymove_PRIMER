import sys
import time
import csv
import datetime
import numpy as np
import os
import socket
import threading
import pyqtgraph as pg
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Slot, Qt

# --- CONFIGURATION GLOBALE ---
FS = 500
CHUNK = 20
NI_DEVICE_NAME = "Dev7"

# --- DRIVERS ---
try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, TerminalConfiguration, LineGrouping
    from nidaqmx.stream_readers import AnalogMultiChannelReader
    from nidaqmx.system import System
    HAS_DAQ = True
    print(">>> PILOTE NIDAQMX ACTIVÉ.")
except ImportError:
    HAS_DAQ = False
    print(">>> MODE SIMULATION ACTIVÉ (nidaqmx absent).")

try:
    from pylsl import StreamInfo, StreamOutlet
    HAS_LSL = True
except ImportError:
    HAS_LSL = False
    print(">>> LSL NON DISPONIBLE.")

# --- RESOLUTION DEVICE NI ---
def pick_ni_device(preferred: str = "Dev7") -> str | None:
    if not HAS_DAQ: return None
    try:
        devs = [d.name for d in System.local().devices]
        if preferred in devs: return preferred
        for d in devs:
            if d.lower().startswith("sim"): return d
        return devs[0] if devs else None
    except: return None

RESOLVED_DEVICE = pick_ni_device(NI_DEVICE_NAME) or NI_DEVICE_NAME
CHAN_ANALOG = f"{RESOLVED_DEVICE}/ai3,{RESOLVED_DEVICE}/ai0,{RESOLVED_DEVICE}/ai1,{RESOLVED_DEVICE}/ai2"
CHAN_VALVE_IN  = f"{RESOLVED_DEVICE}/port0/line0"
CHAN_VALVE_OUT = f"{RESOLVED_DEVICE}/port0/line1"

CALIB_FORCE_A = 149
CALIB_FORCE_B = 7.2
TCP_IP = '127.0.0.1'
TCP_PORT = 65432

# --- STYLE SOMBRE ---
def apply_dark_theme(app):
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.WindowText, Qt.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ToolTipBase, Qt.white)
    palette.setColor(QtGui.QPalette.ToolTipText, Qt.white)
    palette.setColor(QtGui.QPalette.Text, Qt.white)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ButtonText, Qt.white)
    palette.setColor(QtGui.QPalette.BrightText, Qt.red)
    palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    palette.setColor(QtGui.QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)

# --- CONTROLE VANNES ---
def set_valves_state(gonflage=False):
    if not HAS_DAQ: return
    state_in, state_out = (True, False) if gonflage else (False, True)
    try:
        with nidaqmx.Task() as task:
            task.do_channels.add_do_chan(CHAN_VALVE_IN, line_grouping=LineGrouping.CHAN_PER_LINE)
            task.do_channels.add_do_chan(CHAN_VALVE_OUT, line_grouping=LineGrouping.CHAN_PER_LINE)
            task.write([state_in, state_out])
    except Exception as e: print(f"!!! ERREUR VANNES !!! : {e}")

# --- THREAD D'ACQUISITION ---
class AcquisitionThread(QtCore.QThread):
    data_signal = QtCore.Signal(dict)

    def __init__(self):
        super().__init__()
        self.running = True
        self.is_rec = False
        self.csv_writer = None
        self.csv_file = None
        self.target_val = 0.0
        self._target_lock = QtCore.QMutex()
        self._event_label = "Repos"
        self._event_lock = QtCore.QMutex()
        self.tare_val = 0.0

        self.outlets = {}
        if HAS_LSL:
            try:
                self.outlets['force'] = StreamOutlet(StreamInfo('BiodexForce', 'Force', 1, FS, 'float32', 'f01'))
                self.outlets['press'] = StreamOutlet(StreamInfo('OcclusionPressure', 'Pressure', 1, FS, 'float32', 'p01'))
                self.outlets['angle'] = StreamOutlet(StreamInfo('BiodexAngle', 'Angle', 1, FS, 'float32', 'a01'))
                self.outlets['vel']   = StreamOutlet(StreamInfo('BiodexVelocity', 'Velocity', 1, FS, 'float32', 'v01'))
                self.outlets['markers'] = StreamOutlet(StreamInfo('BiodexMarkers', 'Markers', 1, 0, 'string', 'marker01'))
            except Exception as e: print(f"LSL Init Error: {e}")

        self.task = None
        self.data_buffer = np.zeros((4, CHUNK), dtype=np.float64)

        if HAS_DAQ:
            try:
                self.task = nidaqmx.Task()
                self.task.ai_channels.add_ai_voltage_chan(CHAN_ANALOG, terminal_config=TerminalConfiguration.RSE, min_val=-10.0, max_val=10.0)
                self.task.timing.cfg_samp_clk_timing(FS, sample_mode=AcquisitionType.CONTINUOUS)
                self.reader = AnalogMultiChannelReader(self.task.in_stream)
                self.task.start()
            except Exception as e:
                print(f"ERREUR DAQ INIT: {e}")
                self.task = None

    @Slot(str)
    def set_event_label(self, text: str):
        with QtCore.QMutexLocker(self._event_lock): self._event_label = text

    def get_event_label(self):
        with QtCore.QMutexLocker(self._event_lock): return self._event_label

    @Slot(float)
    def set_target(self, value: float):
        with QtCore.QMutexLocker(self._target_lock): self.target_val = value

    def push_marker(self, text: str):
        if HAS_LSL and 'markers' in self.outlets:
            try: self.outlets['markers'].push_sample([text])
            except: pass

    def start_rec(self, filename):
        try:
            self.csv_file = open(filename, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['Timestamp', 'Force_Nm', 'Pressure_mmHg', 'Angle_V', 'Velocity_V', 'Target_Nm', 'Event'])
            self.is_rec = True
        except Exception as e: print(f"CSV Error: {e}")

    def stop_rec(self):
        self.is_rec = False
        if self.csv_file: self.csv_file.close(); self.csv_file = None

    def run(self):
        while self.running:
            if self.task:
                try:
                    self.reader.read_many_sample(self.data_buffer, number_of_samples_per_channel=CHUNK)
                    forces = (CALIB_FORCE_A * self.data_buffer[0, :] + CALIB_FORCE_B) - self.tare_val
                    pressures = (self.data_buffer[1, :] * 0.625 - 0.625) * 750.1
                    angles = self.data_buffer[2, :]
                    velocities = self.data_buffer[3, :]

                    self.data_signal.emit({'force': forces, 'pressure': pressures})

                    if HAS_LSL:
                        self.outlets['force'].push_chunk(forces.reshape(-1, 1).tolist())
                        self.outlets['press'].push_chunk(pressures.reshape(-1, 1).tolist())
                        self.outlets['angle'].push_chunk(angles.reshape(-1, 1).tolist())
                        self.outlets['vel'].push_chunk(velocities.reshape(-1, 1).tolist())

                    if self.is_rec and self.csv_writer:
                        ts0 = time.time()
                        target = self.target_val
                        label = self.get_event_label()
                        rows = [[ts0 + i*(1/FS), forces[i], pressures[i], angles[i], velocities[i], target, label] for i in range(CHUNK)]
                        self.csv_writer.writerows(rows)
                except: time.sleep(0.01)
            else:
                time.sleep(CHUNK/FS)
                f, p = np.random.normal(0, 0.1, CHUNK), np.random.normal(50, 1, CHUNK)
                self.data_signal.emit({'force': f, 'pressure': p})

    def stop(self):
        self.running = False
        if self.task: self.task.close()
        self.wait()

# --- COMMANDES TCP ---
class CommandServer(threading.Thread):
    def __init__(self, interface):
        super().__init__(daemon=True)
        self.interface = interface
        self.running = True

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((TCP_IP, TCP_PORT))
                s.listen()
                while self.running:
                    conn, _ = s.accept()
                    with conn:
                        data = conn.recv(1024)
                        if data:
                            msg = data.decode('utf-8').strip()
                            for m in msg.split('\n'):
                                QtCore.QMetaObject.invokeMethod(self.interface, "handle_remote", Qt.QueuedConnection, QtCore.Q_ARG(str, m))
            except: pass

# --- UI OCCLUSION ---
class TabOcclusion(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__()
        self.interface = parent
        self.occ_timer = QtCore.QTimer()
        self.occ_timer.timeout.connect(self.tick_timer)
        self.setup_ui()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        self.lbl_press_val = QtWidgets.QLabel("0.0 mmHg")
        self.lbl_press_val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_press_val.setStyleSheet("font-size: 80px; font-weight: bold; color: #00e5ff;")
        self.lbl_timer = QtWidgets.QLabel("05:00")
        self.lbl_timer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_timer.setStyleSheet("font-size: 100px; font-weight: bold; color: white;")
        
        btn_start = QtWidgets.QPushButton("DÉMARRER")
        btn_stop = QtWidgets.QPushButton("STOP 🛑")
        btn_start.setMinimumHeight(60)
        btn_stop.setMinimumHeight(60)
        btn_start.clicked.connect(self.start_occlusion)
        btn_stop.clicked.connect(self.stop_occlusion)

        layout.addWidget(self.lbl_press_val)
        layout.addWidget(self.lbl_timer)
        layout.addWidget(btn_start)
        layout.addWidget(btn_stop)

    def start_occlusion(self):
        set_valves_state(True)
        self.end_time = time.time() + 300
        self.occ_timer.start(100)
        self.interface.thread.set_event_label("Start_Occlusion")

    def stop_occlusion(self):
        self.occ_timer.stop()
        set_valves_state(False)
        self.lbl_timer.setText("STOP")
        self.interface.thread.set_event_label("Stop_Occlusion")

    def tick_timer(self):
        rem = self.end_time - time.time()
        if rem <= 0: self.stop_occlusion(); self.lbl_timer.setText("00:00"); return
        m, s = divmod(int(rem), 60)
        self.lbl_timer.setText(f"{m:02d}:{s:02d}")

    def update_display(self, p): self.lbl_press_val.setText(f"{p:.1f} mmHg")

# --- UI FORCE ---
class TabForce(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__()
        self.interface = parent
        self.mvc_max_ref = 1.0
        self.display_frozen = False
        self.is_mvc_recording = False
        self.mvc_buffer = []
        self.timer_countdown = QtCore.QTimer()
        self.timer_countdown.timeout.connect(self.update_countdown_label)
        self.x_buf = np.linspace(-10, 0, FS * 10)
        self.y_buf = np.zeros(FS * 10)
        self.setup_ui()

    def setup_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()

        # Bloc MVC Manuelle
        group_mvc = QtWidgets.QGroupBox("Configuration MVC")
        vmvc = QtWidgets.QVBoxLayout(group_mvc)
        vmvc.addWidget(QtWidgets.QLabel("MVC Max (Nm):"))
        self.spin_mvc = QtWidgets.QDoubleSpinBox()
        self.spin_mvc.setRange(0.1, 1000.0)
        self.spin_mvc.setValue(100.0)
        self.spin_mvc.setSuffix(" Nm")
        self.spin_mvc.valueChanged.connect(self.on_manual_mvc_change)
        vmvc.addWidget(self.spin_mvc)
        left.addWidget(group_mvc)

        # Cibles
        group_tgt = QtWidgets.QGroupBox("Cibles Visuelles")
        vtgt = QtWidgets.QVBoxLayout(group_tgt)
        self.chk_line30 = QtWidgets.QCheckBox("30 % MVC")
        self.chk_line50 = QtWidgets.QCheckBox("50 % MVC")
        self.chk_line30.stateChanged.connect(self.on_line30_toggle)
        self.chk_line50.stateChanged.connect(self.on_line50_toggle)
        vtgt.addWidget(self.chk_line30)
        vtgt.addWidget(self.chk_line50)
        left.addWidget(group_tgt)

        # Marqueurs Concentrique / Excentrique
        group_markers = QtWidgets.QGroupBox("Mode Contraction")
        vmark = QtWidgets.QVBoxLayout(group_markers)
        self.chk_con = QtWidgets.QCheckBox("Concentrique")
        self.chk_ecc = QtWidgets.QCheckBox("Excentrique")
        self.chk_con.stateChanged.connect(self.toggle_con)
        self.chk_ecc.stateChanged.connect(self.toggle_ecc)
        vmark.addWidget(self.chk_con)
        vmark.addWidget(self.chk_ecc)
        left.addWidget(group_markers)

        self.pbar = QtWidgets.QProgressBar()
        self.pbar.setRange(0, 100)
        left.addWidget(self.pbar)

        btn_tare = QtWidgets.QPushButton("⚖️ FAIRE TARE")
        btn_tare.clicked.connect(self.interface.do_tare_global)
        left.addWidget(btn_tare)
        left.addStretch()
        layout.addLayout(left, 1)

        right = QtWidgets.QVBoxLayout()
        self.info_lbl = QtWidgets.QLabel("MODE FORCE")
        self.info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.val_lbl = QtWidgets.QLabel("0.0 Nm")
        self.val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.val_lbl.setStyleSheet("font-size: 50px; color: #00e5ff; font-weight: bold;")
        
        self.plot = pg.PlotWidget()
        self.curve = self.plot.plot(pen=pg.mkPen('#00e5ff', width=3))
        self.line_tgt = pg.InfiniteLine(angle=0, pen=pg.mkPen('#ff1744', width=2, style=Qt.PenStyle.DashLine))
        self.line_30 = pg.InfiniteLine(angle=0, pen=pg.mkPen('#f1c40f', width=2, style=Qt.PenStyle.DashLine))
        self.line_50 = pg.InfiniteLine(angle=0, pen=pg.mkPen('#9b59b6', width=2, style=Qt.PenStyle.DashLine))
        for l in [self.line_tgt, self.line_30, self.line_50]: 
            l.setVisible(False); self.plot.addItem(l)
        self.line_tgt.setVisible(True)

        right.addWidget(self.info_lbl); right.addWidget(self.val_lbl); right.addWidget(self.plot)
        layout.addLayout(right, 4)

    
    def on_line30_toggle(self, state):
        self.update_target_lines()
    # triggers LSL
        if state == Qt.Checked:
            self.interface.thread.push_marker("LINE_30_ON")
            self.interface.thread.set_event_label("Line30_On")
        else:
            self.interface.thread.push_marker("LINE_30_OFF")
            self.interface.thread.set_event_label("Line30_Off")

    def on_line50_toggle(self, state):
        self.update_target_lines()
    # triggers LSL
        if state == Qt.Checked:
            self.interface.thread.push_marker("LINE_50_ON")
            self.interface.thread.set_event_label("Line50_On")
        else:
            self.interface.thread.push_marker("LINE_50_OFF")
            self.interface.thread.set_event_label("Line50_Off")

    def toggle_con(self, state):
        if state:
            self.chk_ecc.setChecked(False)
            self.interface.thread.set_event_label("Concentrique")
            self.interface.thread.push_marker("CON")
        else:
            if not self.chk_ecc.isChecked():
                self.interface.thread.set_event_label("Repos")

    def toggle_ecc(self, state):
        if state:
            self.chk_con.setChecked(False)
            self.interface.thread.set_event_label("Excentrique")
            self.interface.thread.push_marker("ECC")
        else:
            if not self.chk_con.isChecked():
                self.interface.thread.set_event_label("Repos")

    def on_manual_mvc_change(self, val):
        self.mvc_max_ref = val
        self.update_target_lines()

    def update_graph(self, forces):
        m = float(np.mean(forces))
        if not self.display_frozen: self.val_lbl.setText(f"{m:.1f} Nm")
        n = len(forces)
        self.y_buf[:-n] = self.y_buf[n:]; self.y_buf[-n:] = forces
        self.curve.setData(self.x_buf, self.y_buf)
        if self.mvc_max_ref > 0:
            self.pbar.setValue(int(max(0, min(100, (m / self.mvc_max_ref) * 100))))
        if self.is_mvc_recording: self.mvc_buffer.extend(forces)

    def update_target_lines(self):
        self.line_30.setPos(0.30 * self.mvc_max_ref)
        self.line_30.setVisible(self.chk_line30.isChecked())
        self.line_50.setPos(0.50 * self.mvc_max_ref)
        self.line_50.setVisible(self.chk_line50.isChecked())

    def start_local_timer(self, dur, txt, col):
        self.timer_prefix = txt
        self.info_lbl.setStyleSheet(f"color: {col}; font-size: 30px; font-weight: bold;")
        self.target_time = time.time() + dur
        self.timer_countdown.start(100)

    def update_countdown_label(self):
        rem = max(0, self.target_time - time.time())
        self.info_lbl.setText(f"{self.timer_prefix} ({int(rem)}s)")
        if rem == 0: self.timer_countdown.stop()

# --- MAIN INTERFACE ---
class BioMasterInterface(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('BIODEX MASTER - SOMBRE')
        self.resize(1100, 750)
        self.tabs = QtWidgets.QTabWidget()
        self.tab_occ = TabOcclusion(self)
        self.tab_force = TabForce(self)
        self.tabs.addTab(self.tab_occ, "🛑 OCCLUSION")
        self.tabs.addTab(self.tab_force, "💪 FORCE")
        self.setCentralWidget(self.tabs)

        tbar = self.addToolBar("REC")
        self.input_sub = QtWidgets.QLineEdit("S01")
        self.btn_rec = QtWidgets.QPushButton("🔴 REC")
        self.btn_rec.setCheckable(True)
        self.btn_rec.clicked.connect(self.toggle_rec)
        tbar.addWidget(QtWidgets.QLabel(" ID: ")); tbar.addWidget(self.input_sub); tbar.addSeparator(); tbar.addWidget(self.btn_rec)

        self.thread = AcquisitionThread()
        self.thread.data_signal.connect(self.dispatch_data)
        self.thread.start()
        self.server = CommandServer(self); self.server.start()

    def toggle_rec(self):
        if self.btn_rec.isChecked():
            fname = os.path.join(os.getcwd(), f"DATA_{self.input_sub.text()}_{time.strftime('%H%M%S')}.csv")
            self.thread.start_rec(fname)
            self.btn_rec.setText("STOP REC"); self.btn_rec.setStyleSheet("background-color: red;")
        else:
            self.thread.stop_rec(); self.btn_rec.setText("🔴 REC"); self.btn_rec.setStyleSheet("")

    def do_tare_global(self):
        # On calcule la moyenne de la force brute actuelle et on l'ajoute à la tare existante
        current_raw_mean = np.mean(self.thread.data_buffer[0, :] * CALIB_FORCE_A + CALIB_FORCE_B)
        self.thread.tare_val = current_raw_mean
        print(f"Tare effectuée: {self.thread.tare_val:.2f} Nm")

    @Slot(dict)
    def dispatch_data(self, data):
        self.tab_occ.update_display(float(np.mean(data['pressure'])))
        self.tab_force.update_graph(data['force'])

    @Slot(str)
    def handle_remote(self, msg):
        if msg == "TARE": self.do_tare_global()
        elif msg.startswith("TIMER_START:"):
            p = msg.split(":")
            self.tab_force.start_local_timer(float(p[1]), p[2], p[3] if len(p)>3 else "white")
        elif msg.startswith("TARGET:"):
            val = float(msg.split(":")[1]) * self.tab_force.mvc_max_ref / 100
            self.tab_force.line_tgt.setPos(val); self.tab_force.line_tgt.setVisible(True)
            self.thread.set_target(val)
        elif msg == "MVC_START":
            self.tab_force.mvc_buffer = []; self.tab_force.is_mvc_recording = True
            self.tab_force.info_lbl.setText("REC MVC...")
        elif msg == "MVC_STOP":
            self.tab_force.is_mvc_recording = False
            if self.tab_force.mvc_buffer:
                res = float(np.percentile(self.tab_force.mvc_buffer, 98))
                self.tab_force.spin_mvc.setValue(res) 
                self.tab_force.info_lbl.setText(f"MVC: {res:.1f} Nm")

    def closeEvent(self, event):
        set_valves_state(False); self.thread.stop(); event.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    apply_dark_theme(app)
    win = BioMasterInterface()
    win.show()
    sys.exit(app.exec())