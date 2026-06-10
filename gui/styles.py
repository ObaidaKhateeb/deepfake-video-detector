"""
gui/styles.py
Application-wide stylesheet and color constants.
Design: dark forensic theme — near-black base, amber/red accents.
Typography: monospace data, clean sans labels.
"""

# Color tokens
BG_DARK      = "#0e0f13"
BG_CARD      = "#16181f"
BG_ELEVATED  = "#1e2029"
BORDER       = "#2a2d3a"
TEXT_PRIMARY = "#e8eaf0"
TEXT_MUTED   = "#6b7280"
ACCENT       = "#f59e0b"       # amber — analysis in progress
DANGER       = "#ef4444"       # red — fake
WARNING      = "#f97316"       # orange — suspicious
SUCCESS      = "#22c55e"       # green — real
HIGHLIGHT    = "#3b82f6"       # blue — selected / info

# Score color thresholds
def score_color(score: float) -> str:
    if score < 0.35:
        return SUCCESS
    elif score < 0.65:
        return WARNING
    else:
        return DANGER


MAIN_STYLE = f"""
QMainWindow, QWidget {{
    background-color: {BG_DARK};
    color: {TEXT_PRIMARY};
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 13px;
}}

QLabel {{
    color: {TEXT_PRIMARY};
    background: transparent;
}}

QPushButton {{
    background-color: {BG_ELEVATED};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 18px;
    font-size: 13px;
}}

QPushButton:hover {{
    background-color: #252836;
    border-color: {ACCENT};
    color: {ACCENT};
}}

QPushButton:pressed {{
    background-color: #1a1c26;
}}

QPushButton:disabled {{
    color: {TEXT_MUTED};
    border-color: {BORDER};
}}

QPushButton#primary {{
    background-color: {ACCENT};
    color: #0e0f13;
    border: none;
    font-weight: bold;
}}

QPushButton#primary:hover {{
    background-color: #fbbf24;
    color: #0e0f13;
}}

QPushButton#primary:disabled {{
    background-color: #4a3a10;
    color: #7a6a30;
}}

QProgressBar {{
    background-color: {BG_ELEVATED};
    border: 1px solid {BORDER};
    border-radius: 4px;
    height: 8px;
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 3px;
}}

QScrollArea {{
    background-color: transparent;
    border: none;
}}

QScrollBar:vertical {{
    background: {BG_DARK};
    width: 8px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 4px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background: {TEXT_MUTED};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QFrame#card {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}

QFrame#divider {{
    background-color: {BORDER};
    max-height: 1px;
}}
"""
