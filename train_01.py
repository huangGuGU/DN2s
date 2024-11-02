import json
import os

import torch
from utils.dataset_util import dataloader
from utils.device_util import operationdevice
from utils.evalloss_util import eval_loss
from utils.log_util import logger, output_config
from utils.model_util import models
from utils.opt_util import optimizers
from utils.save_util import save_result_image_loss
from utils.trainloss_util import train_loss
from torch.cuda.amp import autocast, GradScaler
import copy
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from skimage.metrics import structural_similarity as compare_ssim

scaler = GradScaler()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_01(m, e, tild, opt, cri, log, dev, cfg,layer):
    m.train()
    total_loss = 0
    samples = 0
    batch_idx = 0
    train_ssim_num = 0
    for batch_idx, (data, target, fn) in enumerate(tild):
        data = data.to(dev)
        target = target.to(dev)
        opt.zero_grad()
        with autocast():
            output = m(data)
            loss = cri(output, target)
            o = copy.deepcopy(output.cpu().detach().numpy()[0][0])
            l = copy.deepcopy(target.cpu().detach().numpy()[0][0])
            train_ssim_num += compare_ssim(o, l)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()


        total_loss += loss.data.item()
        samples += data.shape[0]
        if (batch_idx + 1) % int(cfg['batch_print']) == 0:
            outputstring = 'Train Epoch: {} Batch: [{}~{}/{}], learn_rate:{:.6f} loss:{:.6f},' \
                .format(e, samples - data.shape[0], samples, len(tild.dataset),
                        opt.param_groups[0]['lr'],
                        loss.data.item())
            log.info(outputstring)



    avg_loss = total_loss / (batch_idx + 1)

    train_avg_ssim = train_ssim_num/(batch_idx+1)



    outputstring = 'Train Epoch: {}, Average Loss:{:.6f}'.format(e, avg_loss)
    log.info(outputstring)


    with open(f'train_loss_{layer}.txt', 'a') as fl:
        fl.write('{}:{}:{}\n'.format(e, avg_loss,train_avg_ssim))


def test_01(m, e, tsld, eva, log, dev, cfg,eva2,layer):
    m.eval()
    total_loss = 0
    ssim_num = 0

    with torch.no_grad():
        idx = 0
        for idx, (data, target, fn) in enumerate(tsld):
            data = data.to(dev)
            target = target.to(dev)
            with torch.no_grad():
                with autocast():
                    output = m(data)
            loss = eva(output, target)

            o = copy.deepcopy(output)
            l = copy.deepcopy(target)
            o = o.cpu().detach().numpy()[0][0]
            l = l.cpu().detach().numpy()[0][0]
            ssim_num+= compare_ssim(o,l)

            total_loss += loss.data.item()

            outputstring = 'Test Data: {} Loss: {:.6f}'.format(idx, loss.data.item())
            log.info(outputstring)
            if cfg['save_test_img']:
                path = os.path.join(cfg['test_result_save_dir'],str(layer))
                save_result_image_loss(path, e, fn, output, loss.data.item())


    avg_loss = total_loss / (idx+1)
    avg_ssim = ssim_num/(idx+1)


    outputstring = 'Test Average {}: {:.6f} '.format(eva._get_name(), avg_loss)
    log.info(outputstring)

    save_model(cfg, m, e, avg_loss,layer)



    with open(f'test_loss_{layer}.txt', 'a') as fl:
        fl.write('{}:{}:{}\n'.format(e, avg_loss,avg_ssim))


def adjust_learning_rate(opt, cfg):
    if cfg['scheduler_mode'] == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=cfg['step_size'],
            gamma=cfg['gamma'],
            last_epoch=cfg['last_epoch']
        )
        return lr_scheduler
    elif cfg['scheduler_mode'] == 'multi':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt,
            milestones=cfg['milestones'],
            gamma=cfg['gamma'],
            last_epoch=cfg['last_epoch']
        )
        return lr_scheduler
    elif cfg['scheduler_mode'] == 'exp':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            opt,
            gamma=cfg['gamma'],
            last_epoch=cfg['last_epoch']
        )
        return lr_scheduler
    elif cfg['scheduler_mode'] == 'cos':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=cfg['T_max'],
            eta_min=cfg['eta_min'],
            last_epoch=cfg['last_epoch']
        )
        return lr_scheduler


def save_model(cfg, m, e, loss,layer):
    save_path = os.path.join(cfg['model_save_dir'],str(layer))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if loss is not None:
        torch.save(m.state_dict(), os.path.join(save_path, f'{layer}_{type(m).__name__}_{e}_{loss:.6f}.pth'))
    else:
        torch.save(m.state_dict(), os.path.join(save_path, f'{layer}_{type(m).__name__}_{e}_ending.pth'))


def model_structure_parameters(m, log):
    blank = ' '
    log.info('-' * 119)
    log.info('|' + ' ' * 30 + 'weight name' + ' ' * 30 + '|' + ' ' * 10 + 'weight shape' + ' ' * 10 + '|' + ' ' * 3 +
             'number' + ' ' * 3 + '|')
    log.info('-' * 119)
    num_para = 0

    for index, (key, w_variable) in enumerate(m.named_parameters()):
        if len(key) <= 69:
            key = key + (69 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 30:
            shape = shape + (30 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank
        log.info(f'{index}')
        log.info('| {} | {} | {} |'.format(key, shape, str_num))

    log.info('-' * 119)
    log.info('The total number of parameters: ' + str(num_para))
    log.info('The parameters of Model {}: {:.2f}M'.format(type(m).__name__, num_para / 1e6))
    log.info('-' * 119)


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)
    logger = logger(config['log_filename_train_test'])
    output_config(config, logger)

    device = operationdevice(config, logger)

    trainloader, testloader = dataloader(config)
    for l in [2]:

        model = models(config, device, logger,l)

        model_structure_parameters(model, logger)

        criterion = train_loss(config['loss_function'], logger)

        evaluation = eval_loss('npcc', logger)
        evaluation2 = eval_loss('npcc', logger)

        optimizer = optimizers(config, model, logger)

        scheduler = adjust_learning_rate(optimizer, config)

        for epoch in range(int(config['epochs'])):
            train_01(model, epoch, trainloader, optimizer, criterion, logger, device, config,l)
            test_01(model, epoch, testloader, evaluation, logger, device, config,evaluation2,l)
            scheduler.step()

