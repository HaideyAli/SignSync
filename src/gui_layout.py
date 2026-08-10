"""Widget construction for the SignBridge window.

Split from gui_app.py so that file stays with signal wiring and slots, and to
respect the 150-line rule. build_ui() attaches widgets to the window as
attributes, which is how gui_app's slots reach them.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QListWidget, QPushButton,
                                QTextEdit, QVBoxLayout, QWidget)

def build_ui(self, mode: str) -> None:
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

