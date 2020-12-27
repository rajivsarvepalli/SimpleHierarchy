"""A package for basic hierarchies in pytorch."""
try:  # pragma: no cover
    from importlib.metadata import PackageNotFoundError  # type: ignore
    from importlib.metadata import version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError  # type: ignore
    from importlib_metadata import version


try:  # pragma: no cover
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
