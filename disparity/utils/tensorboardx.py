from tensorboardX import SummaryWriter
import numpy as np
writer = SummaryWriter(log_dir='/disk1/hyj/DFAStereo/ver2.0/runs')
for epoch in range(100):
    writer.add_scalar('/scalar/test',np.random.rand(),epoch)
    writer.add_scalars('/scalar/scalars_test',{'stage0 test':epoch*np.sin(epoch),'stage0 train':epoch*np.cos(epoch),
                                               'stage1 test': epoch * np.sin(epoch)+20,
                                               'stage1 train': epoch * np.cos(epoch)+20},epoch)
writer.close()