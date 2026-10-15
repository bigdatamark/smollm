# [MODULE DOC] Global constants for smolvlm: controller/worker heartbeat timings (seconds), log directory,
# label ignore
# index (-100, skipped by the loss), and the image/video placeholder tokens used in data vs. model prompts.
# Note DEFAULT_VIDEO_TOKEN is "<image>": video frames are fed to the model as image tokens.
CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100

DATA_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_TOKEN = "<image>"

DATA_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_TOKEN = "<image>"