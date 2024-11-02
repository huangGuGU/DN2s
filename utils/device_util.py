# encoding:utf-8
import sys
import torch


def operationdevice(config, logger):
    device_type = config['device_type']
    if device_type == 'cuda':
        cuda_idx = int(config['cuda_idx'])
        if torch.cuda.is_available():
            gpu_num = torch.cuda.device_count()
            if cuda_idx < gpu_num:
                dev = torch.device('cuda:{}'.format(config['cuda_idx']))
            else:
                dev = torch.device('cuda:0')
        else:
            dev = torch.device('cpu')
    elif device_type == 'ddp' :
        dev =  torch.device('cuda')
    else:
        if device_type == 'cpu':
            dev = torch.device('cpu')
            logger.info(f'The model will train on {dev}...')
        else:
            logger.warning(f'The key cuda_idx:{device_type} is invalid !!!!!! ')
            sys.exit()
    return dev
