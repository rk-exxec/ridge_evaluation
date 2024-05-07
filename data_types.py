from dataclasses import dataclass, KW_ONLY, InitVar#
from sklearn.metrics import r2_score
from enum import Enum
import numpy as np
from functools import total_ordering
# from numpy import ndarray, asarray

@total_ordering
class RidgeFitMethod(Enum):
    NONE = 0
    SHANAHAN = 1
    LIMAT = 2
    STYLE_LD = 3
    STYLE = 4
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented
    
    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

@dataclass
class RidgeFit:
    x_values: np.ndarray
    y_values: np.ndarray
    volume: float
    radius: float
    shear_mod: float
    iron_content: int
    gamma: float

    fit_method: RidgeFitMethod
    fit_func: InitVar[callable]

    popt: np.ndarray
    pcov: np.ndarray
    _: KW_ONLY
    perr: np.ndarray = None
    r2: float = None
    # filename: str = None

    # circ_params: np.ndarray = None

    def __post_init__(self, fit_func):
        self.x_values = np.asarray(self.x_values)
        self.y_values = np.asarray(self.y_values)
        self.popt = np.asarray(self.popt)
        if not self.perr and self.pcov is not None: 
            self.perr = np.sqrt(np.diag(self.pcov))
            self.r2 = r2_score(self.y_values, fit_func(self.x_values, self.gamma, self.radius, *self.popt))
        else:
            self.perr = 0
            self.r2 = 1
        


