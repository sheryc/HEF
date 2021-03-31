import argparse
import collections
import time
from functools import partial

import numpy as np
import torch
import transformers
from torch.distributed import destroy_process_group

import data_loader.data_loaders as module_dataloader
import data_loader.dataset as module_dataset
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer


def main(config):
    torch.tensor([1], device=torch.device('cuda'))  # This line is only for occupying gpus earlier >w<
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    logger = config.get_logger('train')

    # setup data_loader instances
    full_dataset = config.initialize('dataset', module_dataset)
    train_data_loader = config.initialize('train_data_loader', module_dataloader, full_dataset,
                                          distributed=config.distributed)
    logger.info(train_data_loader)
    validation_data_loader = config.initialize('validation_data_loader', module_dataloader, full_dataset,
                                               distributed=config.distributed)
    logger.info(validation_data_loader)

    # build model architecture, then print to console
    model = config.initialize('arch', module_arch, full_dataset.emb, full_dataset.max_level * 3)
    logger.info(model)

    # get function handles of loss and metrics
    pathfinder_loss = getattr(module_loss, config['loss']['pathfinder_loss'])
    stopper_loss = getattr(module_loss, config['loss']['stopper_loss'])
    loss = partial(module_loss.mt_loss, pathfinder_loss_fn=pathfinder_loss, stopper_loss_fn=stopper_loss,
                   eta=config['loss']['eta'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    pre_metric = module_metric.obtain_ranks

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    if config['arch']['type'] == 'HEFBert':
        opt_config = config['optimizer']
        initial_lr, init_layer_num, decay_rate, weight_decay = \
            opt_config['args']['lr'], config['arch']['args']['init_layers'], opt_config['decay_rate'], opt_config[
                'weight_decay']
        optimizer = config.initialize('optimizer', torch.optim, model.generate_decayed_param_list(
            initial_lr, init_layer_num, decay_rate, weight_decay))
        # optimizer = config.initialize('optimizer', torch.optim, trainable_params)
    else:
        optimizer = config.initialize('optimizer', torch.optim, trainable_params)
    if config['lr_scheduler']['from'] == 'transformers':
        lr_scheduler = config.initialize('lr_scheduler', transformers, optimizer)
    else:
        lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    start = time.time()
    logger.info('Constructing trainer')
    trainer = Trainer(model, loss, metrics, pre_metric, optimizer, config=config,
                      data_loader=train_data_loader, valid_data_loader=validation_data_loader,
                      lr_scheduler=lr_scheduler)

    logger.info('Begin training')
    trainer.train()
    end = time.time()
    logger.info(f"Finish training in {end - start} seconds")
    if config.distributed:
        destroy_process_group()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='HEF training and validation phase')
    args.add_argument('-c', '--config', required=True, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args.add_argument('-s', '--suffix', default="", type=str, help='suffix indicating this run (default: None)')
    args.add_argument('--seed', default=42, type=int, help='random seed')
    args.add_argument("--local_rank", default=0, type=int)

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--epochs'], type=int, target=('trainer', 'epochs')),
        CustomArgs(['--eta'], type=float, target=('loss', 'eta')),
        CustomArgs(['--lr'], type=float, target=('optimizer', 'args', 'lr'))
    ]
    config = ConfigParser(args, options)
    main(config)
