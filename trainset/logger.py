import os
import re
import sys
import time
import json
import torch
import pickle
import random
import logging
import argparse
import subprocess
import numpy as np
from datetime import timedelta, date
from .utils import get_code_version


class LogFormatter(logging.Formatter):
    """Custom log formatter to add elapsed time since the start."""

    def __init__(self):
        super().__init__()
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)
        prefix = f"{record.levelname} - {time.strftime('%x %X')} - {timedelta(seconds=elapsed_seconds)}"
        message = record.getMessage().replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return f"{prefix} - {message}" if message else ''


def create_logger(filepath=None, rank=0):
    """
    Create and configure a logger.
    :param filepath: File path for saving logs (optional).
    :param rank: Process rank (used for multi-GPU training).
    :return: Configured logger instance.
    """
    log_formatter = LogFormatter()

    logger = logging.getLogger()
    logger.handlers = []  # Clear any existing handlers
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # File handler for logging to a file
    if filepath:
        if rank > 0:
            filepath = f"{filepath}-{rank}"
        file_handler = logging.FileHandler(filepath, "a", encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)

    # Console handler for logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    # Add a reset function to the logger to reset the elapsed time
    logger.reset_time = lambda: setattr(log_formatter, 'start_time', time.time())

    return logger


def initialize_exp(params):
    """
    Initialize the experiment:
    - Create a folder to store results
    - Log experiment parameters and configurations
    :param params: Experiment parameters (argparse.Namespace).
    :return: (logger, experiment folder path)
    """
    exp_folder = get_dump_path(params)

    # Save parameters to a file
    params_path = os.path.join(exp_folder, 'params.pkl')
    with open(params_path, 'w') as f:
        json.dump(vars(params), f, indent=4)

    # Construct and log the command used to run the script
    command = construct_command(params)
    params.command = f"{command} --exp_id \"{params.exp_id}\""

    # Ensure experiment name is valid
    assert params.exp_name.strip(), "Experiment name cannot be empty."

    # Create and configure the logger
    logger = create_logger(os.path.join(exp_folder, 'train.log'), rank=getattr(params, 'global_rank', 0))
    log_experiment_details(logger, params, exp_folder, command)

    return logger, exp_folder


def construct_command(params):
    """Construct the command string used to run the experiment."""
    command = ["python", sys.argv[0]]
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            assert '"' not in arg and "'" not in arg
            command.append(arg)
        else:
            assert "'" not in arg
            formatted_arg = f"'{arg}'" if not re.match('^[a-zA-Z0-9_]+$', arg) else arg
            command.append(formatted_arg)
    return ' '.join(command)


def log_experiment_details(logger, params, exp_folder, command):
    """Log experiment details to the logger."""
    logger.info("============ Initialized logger ============")
    for k, v in sorted(vars(params).items()):
        logger.info(f"{k}: {v}")
    logger.info(f"# Git Version: {get_code_version()} #")
    logger.info(f"The experiment will be stored in {exp_folder}")
    logger.info(f"Running command: {command}\n")


def get_dump_path(params):
    """
    Create the experiment directory if it doesn't exist.
    :param params: Experiment parameters (argparse.Namespace).
    :return: Experiment folder path.
    """
    assert params.exp_name, "Experiment name cannot be empty."
    assert params.dump_path, "Please specify a dump path."

    # Create a directory with today's date as a prefix
    date_prefix = date.today().strftime('%m%d-')
    sweep_path = os.path.join(params.dump_path, f"{date_prefix}{params.exp_name}")
    os.makedirs(sweep_path, exist_ok=True)

    # Generate a random experiment ID if not provided
    if not params.exp_id:
        params.exp_id = generate_random_exp_id(sweep_path)

    # Create the experiment folder
    exp_folder = os.path.join(sweep_path, params.exp_id)
    os.makedirs(exp_folder, exist_ok=True)

    return exp_folder


def generate_random_exp_id(sweep_path, length=10):
    """Generate a unique random ID for the experiment."""
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
    while True:
        exp_id = ''.join(random.choice(chars) for _ in range(length))
        if not os.path.isdir(os.path.join(sweep_path, exp_id)):
            return exp_id


if __name__ == '__main__':
    pass
