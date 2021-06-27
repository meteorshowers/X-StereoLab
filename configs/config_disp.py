import os
import numpy as np
from yacs.config import CfgNode as CN

cfg = CN()

cfg.cnt = 0

cfg.btrain = 4


#------------- disparity ---------------#
cfg.model = 'stereonet' # ['stereonet', 'activestereonet', 'hitnet', 'sos']
cfg.maxdisp = 192
cfg.mindisp = 0
cfg.loss_disp = True
#--------------volume--------------------------#
cfg.PlaneSweepVolume = False
cfg.DispVolume = True


#------------- depth ---------------#

#------------- detection ---------------#


#-------------- debug ----------------#
cfg.debug = False

#-------------- Parameters -----------#

#----------------- centerness --------------#

#----------------------------------------------------#







