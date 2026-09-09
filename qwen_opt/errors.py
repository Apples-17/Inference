class UnsupportedPass(RuntimeError):
    """A candidate cannot run on the installed Torch/Transformers/GPU stack."""


class CorrectnessFailure(RuntimeError):
    """A candidate changed greedy token IDs and must not receive speedup credit."""
