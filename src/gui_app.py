"""SignBridge companion app: floating window that sits beside a Zoom call,
recognizes signed words one at a time, and assembles them into a sentence.
No capture-state-machine or Zoom-HTTP logic here — see capture_engine.py
(via gui_worker.py) and zoom_bridge.py.

    python src/gui_app.py --checkpoint checkpoints/best_model_transformer_50_v10.pth
"""
import argparse
import sys

import cv2
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QListWidget,
                                QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget)

from gui_worker import InferenceWorker
from sentence_engine import assemble_sentence
from theme import DARK_QSS
from zoom_bridge import send_to_zoom_chat


class MainWindow(QMainWindow):
    def __init__(self, checkpoint: str, camera: int, auto: bool):
        super().__init__()
        self.setWindowTitle("SignBridge")
        # Stays above other apps, including a shared Zoom window — that's the
        # point of a Zoom-companion panel, not incidental
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.resize(380, 700)

        self.word_history: list[str] = []
        self._build_ui()

        self.worker = InferenceWorker(checkpoint, camera, auto)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.prediction_ready.connect(self._on_prediction)
        self.worker.status_changed.connect(self._on_status)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _build_ui(self) -> None:
        title = QLabel("SignBridge"); title.setObjectName("titleLabel")
        self.status_label = QLabel("starting..."); self.status_label.setObjectName("statusLabel")

        self.preview = QLabel(); self.preview.setFixedHeight(280)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setObjectName("previewLabel")

        capture_btn = QPushButton("Capture Sign  (hold your sign, then click)")
        capture_btn.setObjectName("captureButton")
        capture_btn.clicked.connect(self._on_capture)

        words_label = QLabel("RECOGNIZED WORDS"); words_label.setObjectName("sectionLabel")
        self.word_list = QListWidget()

        sentence_label = QLabel("SENTENCE"); sentence_label.setObjectName("sectionLabel")
        self.sentence_box = QTextEdit(); self.sentence_box.setReadOnly(True)
        self.sentence_box.setFixedHeight(70)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._on_clear)
        zoom_btn = QPushButton("Send to Zoom")
        zoom_btn.setObjectName("zoomButton")
        zoom_btn.clicked.connect(self._on_send_to_zoom)
        button_row = QHBoxLayout()
        button_row.addWidget(clear_btn)
        button_row.addWidget(zoom_btn)

        layout = QVBoxLayout()
        for w in (title, self.preview, self.status_label, capture_btn,
                  words_label, self.word_list, sentence_label, self.sentence_box):
            layout.addWidget(w)
        layout.addLayout(button_row)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    @Slot(object)
    def _on_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        # fromImage() copies the pixel data, so qimg can be discarded safely after this call
        self.preview.setPixmap(QPixmap.fromImage(qimg).scaledToHeight(
            self.preview.height(), Qt.SmoothTransformation))

    @Slot(str, float, list)
    def _on_prediction(self, word: str, conf: float, alts: list) -> None:
        self.word_history.append(word)
        self.word_list.addItem(f"{word}   ({conf*100:.0f}%)")
        self.word_list.scrollToBottom()
        self.sentence_box.setPlainText(assemble_sentence(self.word_history))

    @Slot(str)
    def _on_status(self, status: str) -> None:
        self.status_label.setText(status)

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self.status_label.setText(f"error: {message}")

    def _on_capture(self) -> None:
        self.worker.request_capture()

    def _on_clear(self) -> None:
        self.word_history.clear()
        self.word_list.clear()
        self.sentence_box.clear()

    def _on_send_to_zoom(self) -> None:
        send_to_zoom_chat(self.sentence_box.toPlainText())

    def closeEvent(self, event) -> None:
        self.worker.stop()
        self.worker.wait()
        event.accept()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best_model_transformer_50_v10.pth")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--auto", action="store_true", help="trigger capture on motion instead of the button")
    args = p.parse_args()

    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)
    win = MainWindow(args.checkpoint, args.camera, args.auto)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
