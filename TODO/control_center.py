import sys
import time
import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets, QtCore, QtGui, QtMultimedia
from PySide6.QtCore import Qt
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop

class ControlCenter(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MASTER CONTROL CENTER - LSL")
        self.setMinimumSize(1100, 850)
        self.setStyleSheet("QMainWindow { background-color: #1a1a1a; } QLabel { color: white; }")

        # --- DATA STORAGE ---
        self.best_mvc = 0.0
        self.mvc_trials = []
        self.current_trial_peak = 0.0
        self.current_serie = 0
        self.rep_count = 0
        self.has_crossed_threshold = False
        self.angle_target = 45  # L'angle à atteindre (à ajuster selon tes tests)
        self.angle_rest = 80    # L'angle de retour au repos (à ajuster)
        #Initalisation audio 
        self.sound_go = QtMultimedia.QSoundEffect()
        self.sound_go.setSource(QtCore.QUrl.fromLocalFile("go.wav"))
        self.sound_go.setVolume(1.0)

        self.sound_stop = QtMultimedia.QSoundEffect()
        self.sound_stop.setSource(QtCore.QUrl.fromLocalFile("stop.wav"))
        self.sound_stop.setVolume(1.0)

        # Buffer pour le graphique (affichage de 5 secondes à 100Hz rafraîchissement)
        self.plot_max_points = 500
        self.plot_data = np.zeros(self.plot_max_points)
        
        # --- LSL ---
        info_m = StreamInfo('ManipMarkers', 'Markers', 1, 0, 'string', 'master_ctrl')
        self.marker_outlet = StreamOutlet(info_m)
        self.inlet_force = None
        streams = resolve_byprop('name', 'BiodexData', timeout=2)
        if streams:
            self.marker_outlet = StreamInlet(streams[0])

        self.init_ui()

        # Timers
        self.timer_lsl = QtCore.QTimer()
        self.timer_lsl.timeout.connect(self.update_all)
        self.timer_lsl.start(10) # 100 Hz

        self.timer_chrono = QtCore.QTimer()
        self.timer_chrono.timeout.connect(self.tick)
        self.countdown = 0.0

    def play_go(self):
        if self.sound_go.status() == QtMultimedia.QSoundEffect.Ready:
            self.sound_go.play()

    def play_stop(self):
        if self.sound_stop.status() == QtMultimedia.QSoundEffect.Ready:
            self.sound_stop.play()

     

    def init_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # --- PANNEAU GAUCHE (Contrôles) ---
        left_panel = QtWidgets.QVBoxLayout()
        
        # ID Sujet
        id_box = QtWidgets.QGroupBox("1. SUJET")
        id_box.setStyleSheet("color: #3498db; font-weight: bold;")
        id_lay = QtWidgets.QHBoxLayout(id_box)
        self.input_id = QtWidgets.QLineEdit("SUJET_001")
        btn_id = QtWidgets.QPushButton("VALIDER")
        btn_id.clicked.connect(self.send_id)
        id_lay.addWidget(self.input_id); id_lay.addWidget(btn_id)
        left_panel.addWidget(id_box)

        # Status & Chrono
        status_box = QtWidgets.QGroupBox("2. ÉTAT DU PROTOCOLE")
        status_box.setStyleSheet("color: #f1c40f; font-weight: bold;")
        status_lay = QtWidgets.QVBoxLayout(status_box)
        self.lbl_status = QtWidgets.QLabel("EN ATTENTE DU BIODEX...")
        self.lbl_status.setWordWrap(True)
        self.lbl_chrono = QtWidgets.QLabel("0.0s")
        self.lbl_chrono.setStyleSheet("font-size: 70px; font-family: monospace; color: #f1c40f;")
        status_lay.addWidget(self.lbl_status, alignment=Qt.AlignCenter)
        status_lay.addWidget(self.lbl_chrono, alignment=Qt.AlignCenter)
        left_panel.addWidget(status_box)

        # Boutons Automatisés
        self.btn_mvc = QtWidgets.QPushButton("LANCER CMV AUTO")
        self.btn_mvc.setStyleSheet("background-color: #d35400; color: white; height: 50px; font-weight: bold; border-radius: 5px;")
        self.btn_mvc.clicked.connect(self.start_mvc_automation)
        
        self.btn_proto = QtWidgets.QPushButton("LANCER PROTOCOLE AUTO")
        self.btn_proto.setStyleSheet("background-color: #27ae60; color: white; height: 50px; font-weight: bold; border-radius: 5px;")
        self.btn_proto.clicked.connect(self.start_protocol_automation)
        
        left_panel.addWidget(self.btn_mvc)
        left_panel.addWidget(self.btn_proto)

        left_panel.addSpacing(20)

        # BOUTON STOP URGENCE
        self.btn_stop = QtWidgets.QPushButton("🛑 STOP / RESET")
        self.btn_stop.setStyleSheet("background-color: #c0392b; color: white; height: 80px; font-size: 18px; font-weight: bold; border-radius: 10px;")
        self.btn_stop.clicked.connect(self.emergency_stop)
        left_panel.addWidget(self.btn_stop)

        left_panel.addStretch()
        main_layout.addLayout(left_panel, 1)

        # --- PANNEAU DROIT (Feedback Visuel) ---
        right_panel = QtWidgets.QVBoxLayout()
        
        self.pw = pg.PlotWidget(title="FORCE BIODEX (Nm)")
        self.pw.setBackground('#111')
        self.pw.setYRange(0, 300)
        self.pw.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.pw.plot(pen=pg.mkPen('#e74c3c', width=4))
        
        # Ligne cible
        self.target_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('#f1c40f', width=3, style=Qt.DashLine))
        self.pw.addItem(self.target_line)
        
        # Affichages numériques
        info_lay = QtWidgets.QHBoxLayout()
        self.lbl_force_now = QtWidgets.QLabel("0.0 Nm")
        self.lbl_force_now.setStyleSheet("font-size: 40px; font-weight: bold; color: #e74c3c;")
        self.lbl_mvc_ref = QtWidgets.QLabel("CMV RÉF : --- Nm")
        self.lbl_mvc_ref.setStyleSheet("font-size: 20px; color: #3498db; font-weight: bold;")
        
        info_lay.addWidget(self.lbl_force_now)
        info_lay.addStretch()
        info_lay.addWidget(self.lbl_mvc_ref)
        
        right_panel.addLayout(info_lay)
        right_panel.addWidget(self.pw)
        main_layout.addLayout(right_panel, 2)

    def update_all(self):
        """Récupère la force et met à jour l'affichage"""
        if self.inlet_force is None:
            streams = resolve_byprop('name', 'BiodexData', timeout=0.1)
            if streams:
                self.inlet_force = StreamInlet(streams[0])
                self.lbl_status.setText("SYSTÈME PRÊT")
        else:
            sample, ts = self.inlet_force.pull_sample(timeout=0.0)
            if sample:
                force_val = sample[0]
                self.lbl_force_now.setText(f"{force_val:.1f} Nm")
                
                # Update Plot
                self.plot_data = np.roll(self.plot_data, -1)
                self.plot_data[-1] = force_val
                self.curve.setData(self.plot_data)
                
                # Capture pic MVC
                if "CMV ESSAI" in self.lbl_status.text():
                    if force_val > self.current_trial_peak:
                        self.current_trial_peak = force_val

    def process_biodex_data(self, force, angle):
    # 1. Mise à jour de l'affichage (ton code actuel)
        self.lbl_angle.setText(f"Angle: {angle:.1f}°")
    
    # 2. Logique de comptage des répétitions par l'ANGLE
    # On imagine ici que l'extension diminue la valeur d'angle (ex: de 90° vers 30°)
    
        if not self.has_crossed_threshold:
        # Phase d'effort : est-ce qu'on a atteint la cible ?
            if angle <= self.angle_target:
                self.rep_count += 1
                self.has_crossed_threshold = True # On verrouille
                self.lbl_reps.setText(f"REPS : {self.rep_count} / 10")
                self.play_go() # Le son de validation
            
            # Optionnel : Envoyer un marqueur LSL pour dire "Rep validée"
                self.marker_outlet.push_sample([f"REP_{self.rep_count}_VALIDATED"])

        else:
        # Phase de retour : est-ce qu'on est revenu assez loin pour réarmer ?
            if angle >= self.angle_rest:
                self.has_crossed_threshold = False # On déverrouille pour la prochaine rep

    def emergency_stop(self):
        """Arrête tout immédiatement et réinitialise"""
        self.play_stop()
        self.timer_chrono.stop()
        self.countdown = 0
        self.lbl_chrono.setText("0.0s")
        self.lbl_status.setText("ARRÊT D'URGENCE / RESET")
        self.target_line.setValue(0)
        self.marker_outlet.push_sample(["EMERGENCY_STOP"])
        self.btn_mvc.setEnabled(True)
        self.btn_proto.setEnabled(True)

    def send_id(self):
        self.marker_outlet.push_sample([f"ID:{self.input_id.text()}"])
        self.input_id.setEnabled(False)

    # --- LOGIQUE AUTOMATISÉE ---
    def start_mvc_automation(self):
        self.btn_mvc.setEnabled(False)
        self.mvc_trials = []
        self.run_mvc_step(1)

    def run_mvc_step(self, n):
        if n > 3:
            self.best_mvc = np.mean(self.mvc_trials)
            self.lbl_mvc_ref.setText(f"CMV RÉF : {self.best_mvc:.1f} Nm")
            self.lbl_status.setText("CMV TERMINÉE")
            self.play_stop()
            self.btn_mvc.setEnabled(True)

            self.marker_outlet.push_sample([f"MVC_VALUE:{self.best_mvc:.2f}"])
            return
        
    
        self.lbl_status.setText(f"CMV ESSAI {n}/3 : CONTRACTEZ !")
        self.play_go()

        self.marker_outlet.push_sample([f"START_MVC_TRIAL_{n}"])
        self.current_trial_peak = 0
        self.start_timer(5.0, lambda: self.end_mvc_step(n))

    def end_mvc_step(self, n):
        self.mvc_trials.append(self.current_trial_peak)
        self.play_stop()
        if n < 3:
            self.lbl_status.setText(f"REPOS CMV ({n}/3)")
            self.start_timer(10.0, lambda: self.run_mvc_step(n+1))
        else:
            self.run_mvc_step(4)

    def start_protocol_automation(self):
        if self.best_mvc <= 0:
            self.lbl_status.setText("⚠️ ERREUR : FAITES LA CMV !")
            return
        self.current_serie = 1
        self.start_new_serie()

    def start_new_serie(self):
        level = 0.3 if self.current_serie <= 3 else 0.5
        target = self.best_mvc * level
        self.target_line.setValue(target)

        self.play_go()

        self.lbl_status.setText(f"SÉRIE {self.current_serie}/6 ({int(level*100)}%)\nCIBLE : {target:.1f} Nm")
        self.marker_outlet.push_sample([f"START_SERIE_{self.current_serie}"])

        self.current_rep = 0
        self.is_rununing_protocol = True

        self.rep_count = 0
        self.has_crossed_threshold = False
        self.lbl_reps.setText("REPS : 0 / 10")

    def start_timer(self, sec, callback):
        self.countdown = sec
        self.timer_callback = callback
        self.timer_chrono.start(100)

    def tick(self):
        if self.countdown > 0:
            self.countdown -= 0.1
            self.lbl_chrono.setText(f"{self.countdown:.1f}s")
        else:
            self.timer_chrono.stop()
            if hasattr(self, 'timer_callback'): self.timer_callback()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = ControlCenter()
    win.show()
    sys.exit(app.exec())