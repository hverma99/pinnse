"""Support for names deprecated in pinnse 1.0 and kept until their removal."""

import warnings

REMOVAL_VERSION = "2.0"


def warn_renamed(old: str, new: str, stacklevel: int = 3) -> None:
    """
    Warn that `old` was renamed to `new`.

    The default `stacklevel` attributes the warning to the caller of the
    function or method that calls this helper.
    """
    warnings.warn(
        f"{old} is deprecated and will be removed in pinnse {REMOVAL_VERSION}; "
        f"use {new} instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
