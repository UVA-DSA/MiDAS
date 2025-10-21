import argparse
import os

from .utils.config import load_config
from .utils.logger import setup_logger


def run_preprocess(cfg_path: str):
	import sys
	from . import preprocess as pp
	# Set up sys.argv for the preprocess module
	original_argv = sys.argv[:]
	sys.argv = ['preprocess.py', '--config', cfg_path]
	try:
		pp.main()
	finally:
		sys.argv = original_argv


def run_train(cfg_path: str):
	import sys
	from . import train as tr
	# Set up sys.argv for the train module
	original_argv = sys.argv[:]
	sys.argv = ['train.py', '--config', cfg_path]
	try:
		tr.main()
	finally:
		sys.argv = original_argv


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, required=True)
	args = parser.parse_args()
	cfg = load_config(args.config)
	paths = cfg["paths"]
	logger = setup_logger("main", paths["logs_dir"], filename="main.log")

	logger.info("Starting preprocessing...")
	run_preprocess(args.config)
	logger.info("Preprocessing done. Starting training...")
	run_train(args.config)

	best_ckpt_txt = os.path.join(paths["ckpt_dir"], "best_checkpoint.txt")
	if os.path.exists(best_ckpt_txt):
		with open(best_ckpt_txt, "r", encoding="utf-8") as f:
			logger.info(f"Best checkpoint: {f.read().strip()}")
	else:
		logger.warning("Best checkpoint not found. Check training logs.")


if __name__ == "__main__":
	main()
