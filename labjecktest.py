# labjack_u3_live_plot.py
"""Python GUI to stream AIN0 from a LabJack U3 at 25 kS/s for 2 s and plot live

Requirements (tested on Windows 7):
    pip install PyQt5==5.15.4 pyqtgraph==0.13.3 numpy LabJackPython

Make sure the **LabJack UD driver** is installed so the `u3` module can see the
hardware. Run the script and a window will plot AIN0 in real‑time for two
seconds and then stop automatically.
"""

import sys
import time
from typing import List

import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

try:
    import u3  # LabJack U3 driver (LabJackPython package)
except ImportError as e:
    raise SystemExit(
        "LabJackPython not found. Install with `pip install LabJackPython`."
    ) from e


class StreamWorker(QtCore.QThread):
    """Background thread that streams data from the U3 and emits it in chunks."""

    new_data = QtCore.pyqtSignal(object)  # Emits numpy.ndarray
    error = QtCore.pyqtSignal(str)

    def __init__(self, *, runtime_sec: float = 2.0, scan_rate: int = 25_000):
        super().__init__()
        self.runtime_sec = runtime_sec
        self.scan_rate = scan_rate
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------
    def run(self) -> None:  # noqa: D401
        device = None
        stream_started = False
        try:
            device = u3.U3()  # Open the first available U3
            # Configure single‑ended stream on AIN0 (Ground = channel 31)
            device.streamConfig(
                NumChannels=1,
                PChannels=[0],  # AIN0
                NChannels=[31],  # 31 = single‑ended
                ScanFrequency=self.scan_rate,
            )
            device.streamStart()  # <‑‑ START STREAMING
            stream_started = True

            start_time = time.perf_counter()
            chunk: List[float] = []
            emit_every_n = int(self.scan_rate * 0.02)  # ≈ every 20 ms

            for packet in device.streamData():  # Generator of result dicts
                if packet is None:
                    continue  # Skip keep‑alives / errors
                ain_vals = packet.get("AIN0")
                if ain_vals:
                    chunk.extend(ain_vals)

                # Push data to the GUI in small slices to keep UI snappy
                if len(chunk) >= emit_every_n:
                    self.new_data.emit(np.asarray(chunk, dtype=np.float32))
                    chunk.clear()

                if self._stop_requested or (time.perf_counter() - start_time) >= self.runtime_sec:
                    break
        except Exception as exc:  # pylint: disable=broad-except
            # Surface errors to the GUI so the user sees them
            self.error.emit(str(exc))
        finally:
            if device is not None:
                # Clean shutdown no matter what
                if stream_started:
                    try:
                        device.streamStop()
                    except Exception:  # Ignore double‑stop or other low‑level errors
                        pass
                device.close()
            # Flush any remaining samples so the plot shows all data
            if "chunk" in locals() and chunk:
                self.new_data.emit(np.asarray(chunk, dtype=np.float32))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LabJack U3 – Live AIN0 Plot (25 kS/s)")
        self.resize(900, 500)
        pg.setConfigOptions(antialias=True)

        # ----------------------------  UI  ----------------------------
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setLabel("bottom", "Sample #")
        self._plot_widget.setLabel("left", "Voltage", units="V")
        self._curve = self._plot_widget.plot(pen=pg.mkPen(width=1))

        self.statusBar().showMessage("Streaming…  0.00 s / 2.00 s")

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.addWidget(self._plot_widget)
        self.setCentralWidget(central)

        # --------------------------  Data  ---------------------------
        self._samples = np.empty(0, dtype=np.float32)

        # ------------------------  Worker  ---------------------------
        self._worker = StreamWorker()
        self._worker.new_data.connect(self._on_new_data)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

        # Timer to update the elapsed‑time indicator
        self._elapsed_timer = QtCore.QElapsedTimer()
        self._elapsed_timer.start()
        self._ui_timer = QtCore.QTimer(self)
        self._ui_timer.timeout.connect(self._update_status)
        self._ui_timer.start(200)  # ms

    # ------------------------------------------------------------------
    # Qt slots
    # ------------------------------------------------------------------
    @QtCore.pyqtSlot(object)
    def _on_new_data(self, chunk: np.ndarray) -> None:
        self._samples = np.concatenate((self._samples, chunk))
        self._curve.setData(self._samples)

    @QtCore.pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Stream error", msg)

    @QtCore.pyqtSlot()
    def _on_finished(self) -> None:
        self.statusBar().showMessage(
            f"Acquisition complete – {self._samples.size:.0f} samples"
        )
        self._ui_timer.stop()

    def _update_status(self) -> None:
        elapsed = min(self._elapsed_timer.elapsed() / 1000.0, 2.0)
        self.statusBar().showMessage(f"Streaming…  {elapsed:.2f} s / 2.00 s")

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------
    def closeEvent(self, event):  # noqa: N802
        if self._worker.isRunning():
            self._worker.stop()
            self._worker.wait()
        event.accept()


if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
