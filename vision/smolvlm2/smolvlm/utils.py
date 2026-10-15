# [MODULE DOC] Small smolvlm utilities: semaphore pretty-printing and a rank-0-only print for distributed
# runs.
import datetime
import logging
import logging.handlers
import os
import sys
import torch.distributed as dist

def pretty_print_semaphore(semaphore):
    # [FUNCTION DOC] HELPER: returns "None" for a None semaphore, else "Semaphore(value=<internal _value>,
    # locked=<bool>)".
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


def mprint(*args, **kwargs):
    # [FUNCTION DOC] Print wrapper for distributed training. Reads RANK (default 0) and WORLD_SIZE (default 1)
    # env vars; when
    # WORLD_SIZE > 1 and torch.distributed is initialized, only rank 0 prints, prefixed with
    # "[dist-<rank>-of-<world_size>]"; otherwise prints normally. Returns None.
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and dist.is_initialized():
        if rank == 0:
            return print(f"[dist-{rank}-of-{world_size}]", *args, **kwargs)
        else:
            return
    else:
        return print(*args, **kwargs)