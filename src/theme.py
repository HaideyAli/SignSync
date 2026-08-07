"""Dark QSS theme for the SignBridge companion window."""

DARK_QSS = """
QMainWindow, QWidget {
    background-color: #14161c;
    color: #e6e8ec;
    font-family: "Segoe UI", sans-serif;
    font-size: 13px;
}
QLabel#titleLabel {
    font-size: 16px;
    font-weight: 600;
    color: #ffffff;
}
QLabel#statusLabel {
    font-size: 12px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 8px;
    background-color: #262a34;
}
QLabel#previewLabel {
    background-color: #000000;
    border-radius: 8px;
    border: 1px solid #2c303a;
}
QPushButton {
    background-color: #1f2430;
    color: #e6e8ec;
    border: 1px solid #343a48;
    border-radius: 8px;
    padding: 8px 14px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #2a2f3d;
    border-color: #4a90e2;
}
QPushButton:pressed {
    background-color: #16191f;
}
QPushButton#captureButton {
    background-color: #2e6f4e;
    border-color: #3a8a63;
}
QPushButton#captureButton:hover {
    background-color: #37835d;
}
QPushButton#zoomButton {
    background-color: #2b4f8c;
    border-color: #3a63ad;
}
QPushButton#zoomButton:hover {
    background-color: #345fa8;
}
QListWidget, QTextEdit {
    background-color: #1a1d24;
    border: 1px solid #2c303a;
    border-radius: 8px;
    padding: 6px;
}
QListWidget::item {
    padding: 4px 6px;
    border-radius: 4px;
}
QListWidget::item:selected {
    background-color: #2b4f8c;
}
QLabel#sectionLabel {
    color: #9aa0ab;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding-top: 4px;
}
"""
