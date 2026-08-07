"""SignBridge companion app: floating window beside a Zoom call, publishing a
captioned video feed as a virtual camera Zoom can select.

    python src/gui_app.py --checkpoint checkpoints/best_model_transformer_50_v10.pth

Press Start captioning, then sign freely; recognized words build a sentence.
See docs/DEMO_SCRIPT.md for Zoom setup and demo sentences."""
import argparse
import sys

import cv2
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QListWidget,
                                QMainWindow, QPushButton, QTextEdit, QVBoxLayout,
                                QWidget)

from gui_worker import InferenceWorker
from theme import DARK_QSS


class MainWindow(QMainWindow):
    def __init__(self, checkpoint: str, camera: int, mode: str, use_vcam: bool):
        super().__init__()
        self.setWindowTitle("SignBridge")
        # Stays above other apps including Zoom — the point of a companion panel
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.resize(400, 760)
        self.mode = mode
        self.captioning = False
        self._build_ui(mode)

        self.worker = InferenceWorker(checkpoint, camera, mode, use_vcam)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.word_accepted.connect(self._on_word)
        self.worker.sentence_ready.connect(self._on_sentence)
        self.worker.status_changed.connect(self._on_status)
        self.worker.vcam_status.connect(self._on_vcam_status)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _build_ui(self, mode: str) -> None:
        title = QLabel("SignBridge"); title.setObjectName("titleLabel")
        self.vcam_label = QLabel("starting virtual camera...")
        self.vcam_label.setObjectName("sectionLabel"); self.vcam_label.setWordWrap(True)

        self.preview = QLabel(); self.preview.setFixedHeight(300)
        self.preview.setAlignment(Qt.AlignCenter); self.preview.setObjectName("previewLabel")
        self.status_label = QLabel("starting..."); self.status_label.setObjectName("statusLabel")

        # Cycle mode drives the rhythm; manual/motion keep a per-sign button
        streams = mode in ("live", "cycle")
        self.capture_btn = QPushButton("Start captioning" if streams else "Capture Sign")
        self.capture_btn.setObjectName("captureButton")
        self.capture_btn.clicked.connect(self._on_capture)

        words_label = QLabel("RECOGNIZED WORDS"); words_label.setObjectName("sectionLabel")
        self.word_list = QListWidget()

        sentence_label = QLabel("SENTENCE"); sentence_label.setObjectName("sectionLabel")
        self.sentence_box = QTextEdit(); self.sentence_box.setReadOnly(True)
        self.sentence_box.setFixedHeight(70)

        clear_btn = QPushButton("Clear"); clear_btn.clicked.connect(self._on_clear)
        row = QHBoxLayout(); row.addWidget(clear_btn)

        layout = QVBoxLayout()
        for w in (title, self.vcam_label, self.preview, self.status_label,
                  self.capture_btn, words_label, self.word_list,
                  sentence_label, self.sentence_box):
            layout.addWidget(w)
        layout.addLayout(row)
        container = QWidget(); container.setLayout(layout)
        self.setCentralWidget(container)

    @Slot(object)
    def _on_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        # fromImage() copies, so qimg may be discarded straight after
        self.preview.setPixmap(QPixmap.fromImage(qimg).scaledToHeight(
            self.preview.height(), Qt.SmoothTransformation))

    @Slot(str, float)
    def _on_word(self, word: str, conf: float) -> None:
        self.word_list.addItem(f"{word}   ({conf*100:.0f}%)")
        self.word_list.scrollToBottom()

    @Slot(str)
    def _on_sentence(self, sentence: str) -> None:
        self.sentence_box.setPlainText(sentence)

    @Slot(str)
    def _on_status(self, status: str) -> None:
        self.status_label.setText(status)

    @Slot(bool, str)
    def _on_vcam_status(self, available: bool, detail: str) -> None:
        self.vcam_label.setText(f"VIRTUAL CAMERA: {detail}" if available
                                else f"VIRTUAL CAMERA OFF — {detail}")

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self.status_label.setText(f"error: {message}")

    def _on_capture(self) -> None:
        """Live/cycle toggle captioning; manual and motion capture one sign."""
        if self.mode not in ("live", "cycle"):
            return self.worker.request_capture()
        self.captioning = not self.captioning
        self.worker.set_running(self.captioning)
        self.capture_btn.setText("Stop captioning" if self.captioning
                                 else "Start captioning")

    def _on_clear(self) -> None:
        self.worker.request_clear()
        self.word_list.clear(); self.sentence_box.clear()

    def closeEvent(self, event) -> None:
        # Bounded wait: the worker can be inside a blocking camera call when
        # stop() lands, and an unbounded wait() here freezes the whole UI until
        # that returns. Terminate as a last resort so closing always works.
        self.worker.stop()
        if not self.worker.wait(3000):
            self.worker.terminate()
            self.worker.wait(1000)
        event.accept()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best_model_transformer_50_v10.pth")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--mode", default="live", choices=["live", "cycle", "manual"],
                   help="live: sign freely, words appear (default); cycle: sign "
                        "to a countdown; manual: one sign per button press")
    p.add_argument("--no-vcam", action="store_true", help="window only, no virtual camera")
    args = p.parse_args()

    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)
    win = MainWindow(args.checkpoint, args.camera,
                     mode=args.mode, use_vcam=not args.no_vcam)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
