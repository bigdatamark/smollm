# [MODULE DOC] Script entry point: when run as __main__, calls smolvlm.train.train.train() to launch training.
from smolvlm.train.train import train

if __name__ == "__main__":
    train()