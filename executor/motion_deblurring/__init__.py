import os

from ..tool import Tool
from ..multitask_tools import *


__all__ = ['motion_deblurring_toolbox']


subtask = 'motion_deblurring'
motion_deblurring_toolbox = [
    Restormer(subtask=subtask), 
    MPRNet(subtask=subtask),
    # MAXIM(subtask=subtask),  # CPU慢，暂时排除
    XRestormer(subtask=subtask),
]