import json
import torch
from utils.dataset_util import dataloader
from utils.device_util import operationdevice
from utils.evalloss_util import eval_loss
from utils.log_util import logger, output_config
from utils.model_util import models
from utils.save_util import save_result_image_loss, create_eval_dir, save_results_in_file
from utils.tensorboard_util import Visualizer


def eval_01(model, evalloader, evaluation, logger, dev, config):
    model.eval()
    total_loss = 0
    fn_list = []
    loss_list = []
    save_dir = create_eval_dir(config['eval_result_save_dir'])
    with torch.no_grad():
        for idx, (data, target, fn) in enumerate(evalloader):
            target = target.to(dev)
            data = data.to(dev)
            output = model(data, target)[:, 0, :, :].unsqueeze(1)
            eval_loss = evaluation(output, target)
            total_loss += eval_loss.data.item()
            fn_list.append(fn)
            loss_list.append(eval_loss.data.item())
            save_result_image_loss(save_dir, None, fn, output, eval_loss.data.item())
            Visual.vis_write('evalloss', {
                config['eval_loss']: eval_loss.data.item(),
            }, idx)
    save_results_in_file(save_dir, config['text_save_filename'], fn_list, loss_list)
    avg_loss = total_loss / len(evalloader.dataset)
    outputstring = 'Eval Average {}: {:.6f} '.format(evaluation._get_name(), avg_loss)
    logger.info(outputstring)


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)

    logger = logger(config['log_filename_eval'])

    output_config(config, logger)

    Visual = Visualizer(config['visualization_eval'])

    device = operationdevice(config, logger)

    _, evalloader = dataloader(config)

    model = models(config, device, logger)

    evaluation = eval_loss(config['eval_loss'], logger)

    eval_01(model, evalloader, evaluation, logger, device, config)

    Visual.close_vis()

