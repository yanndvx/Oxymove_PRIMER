import sys
import time
import datetime
import csv
import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets, QtCore
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop

# --- CONFIGURATION MATÉRIELLE ---
FS = 500
CHUNK = 20
CALIB_FORCE_A = 147.7
CALIB_FORCE_B = 0.5
CALIB_ANGLE_A = -34.54
CALIB_ANGLE_B = 100.36

try:
    from PyDAQmx import Task, DAQmx_Val_RSE, DAQmx_Val_Volts, DAQmx_Val_Rising, \
                        DAQmx_Val_ContSamps, DAQmx_Val_GroupByChannel, int32, byref
    HAS_DRIVER = True
except:
    HAS_DRIVER = False

class AcquisitionThread(QtCore.QThread):
    data_signal = QtCore.Signal(object, object)
    def __init__(self):
        super().__init__()
        self.running = True
        self.is_simu = not HAS_DRIVER
        if HAS_DRIVER:
            try:
                self.task = Task()
                self.task.CreateAIVoltageChan("Dev3/ai3", "Force", DAQmx_Val_RSE, -10.0, 10.0, DAQmx_Val_Volts, None)
                self.task.CreateAIVoltageChan("Dev3/ai2", "Angle", DAQmx_Val_RSE, -10.0, 10.0, DAQmx_Val_Volts, None)
                self.task.CfgSampClkTiming("", FS, DAQmx_Val_Rising, DAQmx_Val_ContSamps, CHUNK)
                self.task.StartTask()
            except: self.is_simu = True
        

    def run(self):
        read_buffer = np.zeros(CHUNK * 2, dtype=np.float64)
        while self.running:
            if not self.is_simu:
                try:
                    self.task.ReadAnalogF64(CHUNK, 10.0, DAQmx_Val_GroupByChannel, read_buffer, CHUNK*2, byref(int32()), None)
                    force = (read_buffer[:CHUNK] * CALIB_FORCE_A) + CALIB_FORCE_B
                    angle = (read_buffer[CHUNK:] * CALIB_ANGLE_A) + CALIB_ANGLE_B
                except: continue
            else:
                time.sleep(CHUNK/FS)
                force = np.random.normal(10, 1, CHUNK)
                angle = np.full(CHUNK, 20.0)
            self.data_signal.emit(force, angle)

class BiodexNode(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        # --- 1. DÉFINITION DES VARIABLES (Toujours en premier) ---
        self.subject_id = "INCONNU"
        self.force_offset = 0.0
        self.raw_samples = []
        self.is_taring = False
        self.is_recording = False
        self.current_data = []
        self.plot_data = np.zeros(500)
        

        self.rep_count=0
        self.has_crossed_threshold=False
        self.angle_target = 80
        self.angle_rest = 130


        # --- 2. CRÉATION DE L'INTERFACE (En deuxième) ---
        # Cette fonction va maintenant trouver "self.subject_id" sans problème
        self.init_ui()

        # --- 3. CONFIGURATION DES TIMERS ET LSL (En dernier) ---
        self.lbl_status.setText("SYSTÈME PRÊT")
        
        # Flux LSL
        info = StreamInfo('BiodexData', 'BioSignals', 2, FS, 'float32', 'biodex_001')
        self.outlet = StreamOutlet(info)
        
        info_markers = StreamInfo('BiodexMarkers', 'Markers', 1, 0, 'string', 'biodex_makers_001')
        self.marker_outlet=StreamOutlet(info_markers)
        self.inlet_markers = None
        self.timer_lsl = QtCore.QTimer()
        self.timer_lsl.timeout.connect(self.check_master)
        self.timer_lsl.start(10)
        
        # Lancement de l'acquisition
        self.thread = AcquisitionThread()
        self.thread.data_signal.connect(self.handle_data)
        self.thread.start()

    def init_ui(self):
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        #labels d'info 
        self.lbl_id = QtWidgets.QLabel(f"SUJET : {self.subject_id}")
        self.lbl_id.setStyleSheet("font-size: 16px; font-weight: bold; color: #3498db;")

        self.lbl_status = QtWidgets.QLabel("ATTENTE DU MASTER...")
        self.lbl_status.setStyleSheet("color: #f1c40f; font-weight: bold;")
        
        #Graphique
        self.pw = pg.PlotWidget(title="FORCE LOCALE (NM)")
        self.curve = self.pw.plot(pen=pg.mkPen('#e74c3c', width=2))
        self.pw.setYRange(0, 200)

        #boutons de contrôle
        btn_layout = QtWidgets.QHBoxLayout()

        self.btn_tare= QtWidgets.QPushButton("TARE")
        self.btn_tare.clicked.connect(self.calibrate_tare)
        self.btn_tare.setStyleSheet("height: 40px; background-color: #8e44ad; color: white; font-weight: bold;")

        self.btn_export = QtWidgets.QPushButton("Exporter en CSV")
        self.btn_export.clicked.connect(self.save_to_csv)
        self.btn_export.setStyleSheet("height: 40px; background-color: #27ae60; color: white; font-weight: bold;")

        btn_layout.addWidget(self.btn_tare)
        btn_layout.addWidget(self.btn_export)

        
        layout.addWidget(self.lbl_id)
        layout.addWidget(self.lbl_status)
        layout.addWidget(self.pw)
        layout.addLayout(btn_layout)

        self.setCentralWidget(central)

    def check_master(self):
        if self.inlet_markers is None:
            streams = resolve_byprop('name', 'ManipMarkers', timeout=0.1)
            if streams: self.inlet_markers = StreamInlet(streams[0])
        else:
            sample, ts = self.inlet_markers.pull_sample(timeout=0.0)
            if sample:
                msg = sample[0]
                if msg == "TARE_REQUEST": 
                    self.calibrate_tare()
                if msg.startswith("ID:"):
                    self.subject_id = msg.split(":")[1]
                    self.lbl_id.setText(f"SUJET : {self.subject_id}")
                    self.lbl_status.setText("CONNECTÉ AU MASTER - PRÊT")
                    self.lbl_status.setStyleSheet("color: #3498db;")
                if msg.startswith("MVC_VALUE"):
                    self.mvc_reference=float(msg.split(":")[1])
                elif "START" in msg:
                    self.is_recording = True
                    self.current_data = []
                    self.lbl_status.setText(f"ENREGISTREMENT : {msg}")
                    self.lbl_status.setStyleSheet("color: #2ecc71;")
                elif "STOP" in msg or "EMERGENCY" in msg:
                    self.is_recording = False
                    self.lbl_status.setText("STOPPED")
                    self.lbl_status.setStyleSheet("color: #e74c3c;")

    def handle_data(self, force, angle):
        # 1. Extraction des valeurs brutes
        raw_f = float(np.mean(force)) 
        cur_a = float(angle[-1])      
        
        # 2. GESTION DE LA TARE (L'accumulation des points)
        if self.is_taring:
            self.raw_samples.append(raw_f)
        
        # 3. CALCUL DE LA FORCE NETTE (Valeur brute - Zéro)
        cur_f = raw_f - self.force_offset
        
        # Print de DEBUG
        print(f"DEBUG - Angle: {cur_a:.2f} | Force: {cur_f:.2f} | Reps: {self.rep_count}")

        # 4. LOGIQUES ET AFFICHAGE
        self.handle_repetition_logic(cur_a)
        
        # Envoi LSL
        self.outlet.push_sample([float(cur_f), float(cur_a)])
        
        # Mise à jour graphique
        self.plot_data = np.roll(self.plot_data, -1)
        self.plot_data[-1] = cur_f
        self.curve.setData(self.plot_data)
        
        # --- ENREGISTREMENT DÉTAILLÉ ---
        if self.is_recording:
            ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.current_data.append([
                ts, 
                round(cur_f, 2), 
                round(cur_a, 2), 
                self.lbl_status.text().replace("\n", " "),
                getattr(self, 'mvc_reference', 0.0)
            ])
    def handle_repetition_logic(self, current_angle):
        """Logique de comptage des répétitions basée sur l'angle"""
        # --- CONFIGURATION DES SEUILS ---
        # Si extension = l'angle diminue (ex: de 80 vers 40)
        if not self.has_crossed_threshold:
            # Phase d'effort : on cherche à descendre sous la cible
            if current_angle <= self.angle_target:
                self.rep_count += 1
                self.has_crossed_threshold = True
                self.lbl_status.setText(f"REP VALIDÉE : {self.rep_count} / 10")
                self.lbl_status.setStyleSheet("color: #2ecc71; font-weight: bold;")
                # On joue le son si tu as configuré pygame
                self.play_go() 
                
                self.marker_outlet.push_sample(["REP_DONE"])

                self.lbl_status.setText(f"REP VALIDEE : {self.rep_count}/10")
                QtWidgets.QApplications.processEvents()
        else:
            # Phase de repos : on doit remonter assez haut pour réarmer
            if current_angle >= self.angle_rest:
                self.has_crossed_threshold = False
                self.lbl_status.setText(f"PRÊT POUR REP {self.rep_count + 1}")
                self.lbl_status.setStyleSheet("color: #3498db;")
            


    def save_to_csv(self):
        # Vérification s'il y a des données
        if not self.current_data:
            QtWidgets.QMessageBox.warning(self, "Export", "Aucune donnée enregistrée à exporter.")
            return
            
        # Génération du nom de fichier propre
        t_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"BIODEX_{self.subject_id}_{t_str}.csv"
        
        try:
            # Ouverture et écriture
            with open(fname, 'w', newline='') as f:
                writer = csv.writer(f)
                # Entêtes
                writer.writerow(['Time_Abs', 'Force_Nm', 'Angle_Deg', 'Step', 'MVC_Ref'])
                # Données
                writer.writerows(self.current_data)
                
            QtWidgets.QMessageBox.information(self, "Succès", f"Fichier sauvegardé avec succès :\n{fname}")
            print(f"Export réussi : {fname}")
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Erreur", f"Impossible d'enregistrer le fichier : {str(e)}")

    def calibrate_tare(self):
        """Lance la procédure de tare (moyenne sur 1 seconde)"""
        self.raw_samples = []
        self.is_taring = True
        self.lbl_status.setText("TARE EN COURS... NE PAS TOUCHER")
        self.lbl_status.setStyleSheet("color: #e67e22; font-weight: bold;")
        
        # On attend 1000ms (1s) pour accumuler des données, puis on calcule
        QtCore.QTimer.singleShot(1000, self.finish_tare)

    def finish_tare(self):
        if self.raw_samples:
            self.force_offset = np.mean(self.raw_samples)
            self.is_taring = False
            self.lbl_status.setText("TARE TERMINÉE - ZÉRO FIXÉ")
            self.lbl_status.setStyleSheet("color: #2ecc71; font-weight: bold;")
            print(f"Nouvel offset de force : {self.force_offset:.2f}")
        else:
            self.lbl_status.setText("ERREUR TARE : AUCUNE DONNÉE")
            self.is_taring = False

    

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = BiodexNode()
    win.show()
    sys.exit(app.exec())
