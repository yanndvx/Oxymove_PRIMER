import sys
import time
import numpy as np
import pyqtgraph as pg 
from PySide6 import QtWidgets, QtCore
from PySide6.QtCore import Slot

# --- CONFIG CARTE 2 (BIOPAC) ---
DEVICE_NAME = "Dev2"   # <--- Tu as bien mis Dev5 ici ?
# On appelle bien ai0, ai1, ai2. 
# La carte sait toute seule qu'elle doit aller chercher le "moins" sur ai4, ai5, ai6 car on est en mode 'Diff'.
CHANNELS = f"{DEVICE_NAME}/ai0,{DEVICE_NAME}/ai1,{DEVICE_NAME}/ai2"
FS = 500
CHUNK = 20

# --- DRIVERS ---
try:
    from PyDAQmx import Task, DAQmx_Val_Diff, DAQmx_Val_Volts, DAQmx_Val_Rising, DAQmx_Val_ContSamps, DAQmx_Val_GroupByChannel, int32, byref
    from pylsl import StreamInfo, StreamOutlet
except ImportError:
    print("ERREUR : Installez PyDAQmx et pylsl")
    sys.exit()

class PhysioThread(QtCore.QThread):
    data_signal = QtCore.Signal(object, object, object) 
    
    def __init__(self):
        super().__init__()
        self.running = True
        
        # CRÉATION DU FLUX LSL
        # LabRecorder verra ce flux sous le nom "BiopacPhysio"
        info = StreamInfo('BiopacPhysio', 'Physio', 3, FS, 'float32', 'id_physio_card2')
        # On nomme les canaux pour s'y retrouver plus tard
        chns = info.desc().append_child("channels")
        chns.append_child("channel").append_child_value("label", "ECG_1")
        chns.append_child("channel").append_child_value("label", "ECG_2")
        chns.append_child("channel").append_child_value("label", "EMG_Quad")
        self.outlet = StreamOutlet(info)
        
        self.task = None
        self.buffer = np.zeros(CHUNK * 3, dtype=np.float64)
        
        try:
            self.task = Task()
            self.task.CreateAIVoltageChan(CHANNELS, "", DAQmx_Val_Diff, -10.0, 10.0, DAQmx_Val_Volts, None)
            self.task.CfgSampClkTiming("", FS, DAQmx_Val_Rising, DAQmx_Val_ContSamps, CHUNK)
            self.task.StartTask()
            print(f">>> CARTE BIOPAC ({DEVICE_NAME}) : CONNECTÉE & STREAMING LSL")
        except Exception as e:
            print(f"ERREUR CARTE BIOPAC ({DEVICE_NAME}): {e}")
            self.running = False

    def run(self):
        read_count = int32()
        while self.running:
            if self.task:
                try:
                    # Lecture de la carte
                    self.task.ReadAnalogF64(CHUNK, 10.0, DAQmx_Val_GroupByChannel, self.buffer, CHUNK*3, byref(read_count), None)
                    
                    # Séparation des 3 signaux
                    d1 = self.buffer[0:CHUNK] # ECG 1
                    d2 = self.buffer[CHUNK:CHUNK*2] # ECG 2
                    d3 = self.buffer[CHUNK*2:CHUNK*3] # EMG
                    
                    # Envoi à l'écran pour visu
                    self.data_signal.emit(d1, d2, d3)
                    
                    # Envoi sur le réseau LSL (Pour LabRecorder)
                    # On empile les 3 colonnes pour faire un bloc complet
                    chunk_lsl = np.column_stack((d1, d2, d3))
                    self.outlet.push_chunk(chunk_lsl.tolist())
                    
                except: time.sleep(0.01)
            else:
                time.sleep(0.1)

    def stop(self):
        self.running = False
        if self.task: self.task.StopTask()
        self.wait()

class PhysioInterface(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f'BIOPAC LSL STREAMER ({DEVICE_NAME})')
        self.resize(600, 800)
        self.setStyleSheet("background-color: #222; color: #ccc;")

        self.thread = PhysioThread()
        self.thread.data_signal.connect(self.update)
        self.thread.start()

        w = QtWidgets.QWidget(); self.setCentralWidget(w); layout = QtWidgets.QVBoxLayout(w)
        layout.setSpacing(2)
        
        # 3 Graphiques simples
        self.plots = []
        self.curves = []
        colors = ['#00ff00', '#00ff00', '#ff00ff'] # Vert, Vert, Rose
        names = ["ECG 1", "ECG 2", "EMG"]
        
        for i in range(3):
            layout.addWidget(QtWidgets.QLabel(names[i]))
            p = pg.PlotWidget()
            p.setBackground('k')
            p.showGrid(y=True, alpha=0.3)
            # Optimisation : Downsampling pour ne pas ramer
            p.setDownsampling(mode='peak')
            p.setClipToView(True)
            
            self.plots.append(p)
            self.curves.append(p.plot(pen=colors[i]))
            layout.addWidget(p)

        self.buf_size = FS * 5 # 5 secondes d'affichage
        self.buffers = [np.zeros(self.buf_size) for _ in range(3)]

    def update(self, d1, d2, d3):
        data = [d1, d2, d3]
        for i in range(3):
            num = len(data[i])
            self.buffers[i][:-num] = self.buffers[i][num:]
            self.buffers[i][-num:] = data[i]
            self.curves[i].setData(self.buffers[i], skipFiniteCheck=True)

    def closeEvent(self, e):
        self.thread.stop()
        e.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = PhysioInterface()
    win.show()
    sys.exit(app.exec())
