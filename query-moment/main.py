import torch
import argparse

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("cfg", type=str)
    args.add_argument("--cfg-option", nargs="+")
    args.add_argument("--resume", action="store_true", default=False)
    args.add_argument("--debug", action="store_true", default=False)

def main():
    """resume training if necessary"""