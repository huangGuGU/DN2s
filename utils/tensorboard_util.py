import time
from torch.utils.tensorboard import SummaryWriter


class Visualizer:
    def __init__(self, visual_root, savescalars=True, savegraphs=True):
        self.writer = SummaryWriter(visual_root)
        self.sub_root = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        self.savescalars = savescalars
        self.savegraphs = savegraphs

    def vis_write(self, main_tag, tag_scalar_dict, global_step):
        return self.writer.add_scalars(self.sub_root + '_{}'.format(main_tag),
                                       tag_scalar_dict, global_step)

    def vis_graph(self, model, input_to_model=None):
        with self.writer as w:
            w.add_graph(model, input_to_model)

    def close_vis(self):
        self.writer.close()

