# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.2.24"

import os

# Set ENV Variables (place before imports)
os.environ["OMP_NUM_THREADS"] = "1"  # reduce CPU utilization during training

from ultralytics.data.explorer.explorer import Explorer
from ultralytics.models import NAS, SAM, YOLO, FastSAM
from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "NAS",
    "SAM",
    "FastSAM",
    "checks",
    "download",
    "settings",
    "Explorer",
)
