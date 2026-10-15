# [MODULE DOC] Launcher for step 3 across machines: argv[1] = machine index; shard indices 0..199 split into
# 21 chunks. For
# each shard of this machine's chunk: appends progress lines to /scratch/log.txt, empties /var/log/syslog with
# sudo
# (it fills up during downloads), runs the downloader with "--download_only 1" in its own process group and
# kills
# the whole group with SIGTERM after a 2-hour timeout, then runs it again with "--U 1" to build the image
# dataset
# from whatever was downloaded.
# KNOWN ISSUE: the invoked script m4/sourcing/data_collection/callers/dl_images_create_dataset.py does not
# exist
# in this repo (the equivalent is 03_dl_images_create_dataset.py).
import os
import signal
import subprocess
import sys

import numpy as np


idx_machine = int(sys.argv[1])

IDX_REMAINING = [idx for idx in range(200)]
NUM_MACHINES = 21
IDX = [el.tolist() for el in np.array_split(IDX_REMAINING, NUM_MACHINES)][idx_machine]
PATH_LOG = "/scratch/log.txt"


for idx in IDX:
    f = open(PATH_LOG, "a")
    f.write(f"Starting job {idx}\n")
    f.close()

    os.system("sudo truncate -s 0 /var/log/syslog")

    p = subprocess.Popen(
        f"python3 m4/sourcing/data_collection/callers/dl_images_create_dataset.py {idx} --download_only 1",
        shell=True,
        preexec_fn=os.setsid,
    )
    try:
        p.wait(2 * 60 * 60)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        # p.kill()

    f = open(PATH_LOG, "a")
    f.write(f"{idx} done with download only\n")
    f.close()

    os.system(f"python3 m4/sourcing/data_collection/callers/dl_images_create_dataset.py {idx} --U 1")

    f = open(PATH_LOG, "a")
    f.write(f"{idx} done with create image dataset only\n")
    f.close()
