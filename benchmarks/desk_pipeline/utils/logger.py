import logging
import os
from logging import Logger
from typing import Optional


def setup_logger(name: str, log_dir: str, filename: str = "run.log", level: int = logging.INFO) -> Logger:
	os.makedirs(log_dir, exist_ok=True)
	logger = logging.getLogger(name)
	logger.setLevel(level)
	logger.propagate = False

	# Clear existing handlers to avoid duplication in notebooks / repeated calls
	for h in list(logger.handlers):
		logger.removeHandler(h)

	fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

	file_handler = logging.FileHandler(os.path.join(log_dir, filename))
	file_handler.setFormatter(fmt)
	file_handler.setLevel(level)
	logger.addHandler(file_handler)

	console_handler = logging.StreamHandler()
	console_handler.setFormatter(fmt)
	console_handler.setLevel(level)
	logger.addHandler(console_handler)

	return logger
