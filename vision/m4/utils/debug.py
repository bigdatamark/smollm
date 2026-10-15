# [MODULE DOC] Debug helper for multi-process (multi-GPU) runs: a print replacement serialized via a file
# lock.
import builtins
import fcntl


def printflock(*args, **kwargs):
    # [FUNCTION DOC] Drop-in replacement for print taking the same args/kwargs. Opens this module's own source
    # file for
    # reading, takes an exclusive fcntl flock on it, calls builtins.print, and always releases the lock, so
    # concurrent processes on the same machine never interleave their output. Returns None. Unix-only (fcntl).
    """
    This is a wrapper around the built-in Python `print` which calls `flock` before calling
    `print` and unlocks it immediately after. This wrapper is useful for when each rank needs to
    print a message without getting it interleaved with prints from other ranks.
    The lock file is the file this wrapper is defined in.
    The output order will be random per rank.

    Example:
        >>> # assuming 4 GPUs
        >>> world_size = dist.get_world_size()
        >>> rank = dist.get_rank()
        >>> printflock(f"This is a very long message from rank {rank}/{world_size}")
       This is a very long message from rank 0/4
       This is a very long message from rank 2/4
       This is a very long message from rank 3/4
       This is a very long message from rank 1/4

    It can also be used to override normal `print` for an easier multi-gpu debug:

        from m4.utils.debug import printflock as print

    and then you don't need to change anything in your code, the normal `print` calls will all be non-interleaved
    """

    with open(__file__, "r") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            builtins.print(*args, **kwargs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
