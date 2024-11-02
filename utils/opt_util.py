# encoding:utf-8
import sys
import torch.optim as optim


def optimizers(config, model, logger):
    opt_name = config['optimizer']
    if opt_name == 'sgd':
        return optim.SGD(model.parameters(),
                         lr=config['learning_rate'], momentum=config['momentum'],
                         weight_decay=config['weight_decay'])
    elif opt_name == 'rms':
        return optim.RMSprop(model.parameters(),
                             lr=config['learning_rate'], momentum=config['momentum'],
                             weight_decay=config['weight_decay'])
    elif opt_name == 'adam':
        return optim.Adam(model.parameters(),
                          lr=config['learning_rate'],
                          weight_decay=config['weight_decay'])
    elif opt_name == 'adamw':
        return optim.AdamW(model.parameters(),
                          lr=config['learning_rate'],
                          weight_decay=config['weight_decay'])
    else:
        logger.error('The optimizer name: {} is invalid !!!!!!'
                     .format(config['optimizer']))
        sys.exit()
