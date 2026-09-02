'''Caerno utilities - small helpers for EDA, plotting and feature encoding.

    from caernautilus import imperfection, NanFixer, Digitalize

Submodules stay importable on their own (`from caernautilus import output`),
which is also how the zipped drop-in build is used: `from classes import NanFixer`.
'''

__version__ = "0.2.0"

from .classes import Digitalize, Encoder, FeatureTrans, NanFixer, SlowPolyLinearReg
from .input import number
from .output import (
    imperfection,
    img_framaker,
    img_squeeze,
    informer,
    informer_print,
    multicolumn,
    plot_conf_map,
    plot_some_scatters,
)

__all__ = [
    "Digitalize", "Encoder", "FeatureTrans", "NanFixer", "SlowPolyLinearReg",
    "number",
    "imperfection", "informer", "informer_print", "multicolumn",
    "plot_conf_map", "plot_some_scatters", "img_squeeze", "img_framaker",
]
