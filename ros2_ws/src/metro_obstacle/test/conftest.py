"""Тесты чистых модулей пакета запускаются и без ROS: кладём пакет в sys.path."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
