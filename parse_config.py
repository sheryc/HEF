import logging
import os
from datetime import datetime
from functools import reduce
from operator import getitem
from pathlib import Path

import torch
import torch.distributed as dist

from logger import setup_logging
from utils import read_json, write_json


class ConfigParser:
    def __init__(self, args, options='', timestamp=True, extra_args=None):
        # parse default and custom cli options
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        args = args.parse_args(args=extra_args)

        self.seed = getattr(args, 'seed', 42)

        # setup config json file by -c/--config
        if args.resume:
            self.resume = Path(args.resume)
            self.cfg_fname = self.resume.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            self.resume = None
            self.cfg_fname = Path(args.config)

        # setup device
        # if args.device:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        self.distributed = False
        if 'WORLD_SIZE' in os.environ:
            self.distributed = int(os.environ['WORLD_SIZE']) > 1

        if self.distributed:
            torch.cuda.set_device(args.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')
            self.local_rank = args.local_rank
            self.world_size = max(dist.get_world_size(), 1)
        else:
            self.local_rank = 0
            self.world_size = 1

        # load config file and apply custom cli options
        config = read_json(self.cfg_fname)
        self.__config = _update_config(config, options, args)

        # set save_dir where trained model and log will be saved.

        save_dir = Path(
            ('.' if self.config['trainer']['save_dir'][0] != '.' else '') + self.config['trainer']['save_dir'])
        timestamp = datetime.now().strftime(r'%m%d_%H%M%S') if timestamp else ''
        if "suffix" in args and args.suffix != "":
            timestamp = args.suffix + "_" + timestamp

        experiment_name = self.config['name']

        print(f'Experiment: {experiment_name}')
        print(f'Timestamp for current experiment: {timestamp}')

        if args.resume:
            self.__save_dir = self.resume.parent
            self.__log_dir = self.resume.parent
        else:
            self.__save_dir = save_dir / 'models' / experiment_name / timestamp
            self.__log_dir = save_dir / 'log' / experiment_name / timestamp

            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)

            # save updated config file to the checkpoint dir
            write_json(self.config, self.save_dir / 'config.json')

        # case study during testing phase
        self.case_study = getattr(args, 'case_study', None)

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    def initialize(self, name, module, *args, **kwargs):
        """
        finds a function handle with the name given as 'type' in config, and returns the 
        instance initialized with corresponding keyword args given as 'args'.
        """
        module_cfg = self[name]
        # following is essentially module[module_cfg['type]](*args, **module_cfg['args'])
        return getattr(module, module_cfg['type'])(*args, **module_cfg['args'], **kwargs)

    def __getitem__(self, name):
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity,
                                                                                       self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self.__config

    @property
    def save_dir(self):
        return self.__save_dir

    @property
    def log_dir(self):
        return self.__log_dir


# helper functions used to update config dict with custom cli options
def _update_config(config, options, args):
    for opt in options:
        value = getattr(args, _get_opt_name(opt.flags))
        if value is not None:
            _set_by_path(config, opt.target, value)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
