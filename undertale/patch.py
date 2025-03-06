import os


def patch_filelock_flock_unsupported():
    """Force usage of SoftFileLock by the Datasets library.

    Datasets currently doesn't support filesystems that do not support locking
    (like the SAN filesystem on LLSC). This patches the `filelock` library to
    use a soft lock primitive instead.

    More details: https://github.com/huggingface/datasets/issues/6395.

    Note: The Datasets library can get into some weird states when using soft
    locks - in particular if programs crash in unexpected ways while they own
    locks or if you are attempting to acquire a lock on something that has
    already had a standard flock used on it. You can run the following command
    to clear all locks from the Datasets cache directory:

        rm $(find ~/.cache/huggingface/datasets/ -type f -name '*.lock')

    You may need to run the command above if/when you have switched from flocks
    to soft locks or if a running process with a soft lock dies. Basically, if
    you notice deadlocks, try running the above when you know nothing should
    have those locks (i.e., all processing has stopped).
    """

    import filelock

    filelock.FileLock = filelock.SoftFileLock


if os.environ.get("UNDERTALE_PATCH_FILELOCK_FLOCK_UNSUPPORTED") is not None:
    patch_filelock_flock_unsupported()
