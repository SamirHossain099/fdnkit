"""The package advertises inline type hints to downstream checkers (PEP 561)."""

from importlib.resources import files


def test_py_typed_marker_ships_with_package():
    # Without this file, mypy and pyright ignore every annotation in fdnkit and treat
    # the whole API as Any, so the hints would help nobody outside this repository.
    assert files("fdnkit").joinpath("py.typed").is_file()
