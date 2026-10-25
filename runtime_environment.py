"""Numerical-thread settings shared by the public launchers (standard library only)."""

NUMERICAL_THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"
)


def configure_runtime_environment(dataset, environment):
    """Match the explicit caller setting while preserving unknown inheritance."""
    if dataset == "CIFAR10":
        environment["OMP_NUM_THREADS"] = "4"
    else:
        for name in NUMERICAL_THREAD_VARIABLES:
            environment[name] = "2"
    return environment


def numerical_thread_environment(environment):
    """Return only numerical-thread variables actually present in this mapping."""
    return {name: environment[name] for name in NUMERICAL_THREAD_VARIABLES if name in environment}
