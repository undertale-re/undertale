"""Custom Dask wrappers and utilities."""

from os import listdir
from os.path import basename, join
from typing import Callable, List, Optional

from dask.distributed import Client as DaskClient
from dask.distributed import Future, LocalCluster, WorkerPlugin
from dask_jobqueue import SLURMCluster

from ..logging import get_logger, setup_logging
from ..utils import get_or_create_directory

logger = get_logger(__name__)


def Cluster(
    type: str,
    parallelism: int = 1,
    cores: int = 1,
    memory: str = "16GB",
    partition: Optional[str] = None,
    logs: Optional[str] = None,
):
    """A factory method returning a Dask-compatible Cluster.

    Arguments:
        type: The type of Cluster to create. See ``CLUSTER_TYPES``.
        parallelism: The maximum degree of parallelism.
        cores: The number of cores per worker (SLURM).
        memory: The maximum amount of memory per worker.
        partition: The SLURM partition to use.
        logs: A path to a logging directory - if not provided, the current
            directory will be used.

    Returns:
        An instance of the requested Cluster type. Can be used as a context
        manager.
    """

    cluster: LocalCluster | SLURMCluster

    if type == "local":
        cluster = LocalCluster(
            n_workers=parallelism, threads_per_worker=1, memory_limit=memory
        )
    elif type == "slurm":
        cluster = SLURMCluster(
            queue=partition,
            processes=1,
            cores=cores,
            memory=memory,
            log_directory=logs or "./",
        )
        cluster.scale(jobs=parallelism)
    else:
        raise ValueError(f"{type}: unknown cluster type")

    return cluster


CLUSTER_TYPES = ["local", "slurm"]


class SetupLoggingPlugin(WorkerPlugin):
    def setup(self, worker):
        setup_logging()


def Client(*args, **kwargs):
    """Wraps a Dask Client."""

    client = DaskClient(*args, **kwargs)
    client.register_plugin(SetupLoggingPlugin())

    return client


def merge(client: DaskClient, objects: List) -> Future:
    """Merge a list of objects into a single future.

    This is somehow not functionality that Dask provides natively, but is
    useful if you need to express a joint dependency on several
    objects/futures.

    Arguments:
        client: The Dask Client where tasks should be issued.
        objects: A list of objects to be merged - can be Futures of objects.

    Returns:
        A single Future wrapping the list of objects
    """

    return client.submit(lambda x: x, objects)


def fanout(
    client: DaskClient, function: Callable, inputs: Future, output: str, **kwargs
) -> Future:
    """Executes a given callable across all elements of some iterable Future.

    The given callable should expect a single input file path and generate a
    single output file, returning its path.

    This is equivalent to the following::

        [ function(i, output, **kwargs) for i in inputs ]

    Arguments:
        client: The Dask Client where tasks should be issued.
        function: A callable function to run on each input.
        inputs: A Future containing a list of object over which to run.
        output: The output location where processed results should go.
        **kwargs: Any other keyword arguments to be passed to ``function``.

    Returns:
        A future representing a list of results returned by ``function``
        applied to each of ``inputs``.

    Warning:
        This function will block on ``inputs`` and then materializes it. You
        should probably not plan to pass a lot of data through ``inputs``,
        instead pass a list of filenames to process.
    """

    output, created = get_or_create_directory(output)

    results = []
    if created:
        for input in inputs.result():
            results.append(
                client.submit(function, input, join(output, basename(input)))
            )
    else:
        results = [join(output, f) for f in listdir(output)]

    return merge(client, results)


def flush(client: DaskClient) -> None:
    """Flush the dask client and gracefully shutdown workers.

    This should be run at the end of pipelines to ensure graceful worker
    shutdown and work completion.

    Arguments:
        client: The Dask Client where tasks are run.
    """

    client.retire_workers(close_workers=True, remove=True)
