# encoding:utf-8
import torch
import sys

from models.Onn_Net import Onn_Net
def models(config, device, logger,l):
    if config['model_name'] == 'onn':
        model = Onn_Net(num_layers=l, lam=config['lam'],size=config['size']).to(device)
    else:
        logger.error('{} is invalid!!!'.format(config['model_name']))
        sys.exit()

    def update_model_weight(m, last_weight_dict):
        cur_weight_dict = m.state_dict()
        updated_weight_dict = {k: v for k, v in last_weight_dict.items() if k in cur_weight_dict}
        cur_weight_dict.update(updated_weight_dict)
        m.load_state_dict(cur_weight_dict)

        last_params = len(last_weight_dict)
        cur_params = len(cur_weight_dict)
        matched_params = len(updated_weight_dict)

        infos = [last_params, cur_params, matched_params]
        return m, infos

    logger.info(model)
    model_weight = config['last_model_weight']
    logger.info('Loading last weight from {} for model {}'.format(model_weight, type(model).__name__))

    try:
        model, updata_infos = update_model_weight(model, torch.load(model_weight))
        logger.info('last model params:{}, current model params:{}, matched params:{} !!!!!!'
                    .format(updata_infos[0], updata_infos[1], updata_infos[2]))
    except:
        logger.warning('Can not load last weight from {} for model {} !!!!!!'
                       .format(model_weight, type(model).__name__))
        logger.info('The parameters of model is initialized by method in model set !!!!!!')
    return model
