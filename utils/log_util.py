# encoding:utf-8
import logging


def logger(log_filename):
    log = logging.getLogger()
    log.setLevel(logging.INFO)

    fh = logging.FileHandler(log_filename)
    sh = logging.StreamHandler()

    fm = logging.Formatter('%(asctime)s - %(filename)s [line:%(lineno)d] - %(levelname)s: %(message)s')

    fh.setFormatter(fm)
    sh.setFormatter(fm)

    log.addHandler(fh)
    log.addHandler(sh)

    log.info('------------The Phase will start.....-------------')
    return log


def output_config(cfg, log):
    for key, val in cfg.items():
        log.info(f'{key}:{val}')
