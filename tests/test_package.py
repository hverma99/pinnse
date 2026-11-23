import importlib.resources

import pytest

import pinnse


def test_public_api_is_exported():
    for name in pinnse.__all__:
        assert hasattr(pinnse, name), name


def test_package_ships_type_hints():
    # PEP 561 marker, listed in [tool.setuptools.package-data]
    assert importlib.resources.files("pinnse").joinpath("py.typed").is_file()


def test_dataloader_alias_is_deprecated_torch_dataloader():
    from torch.utils.data import DataLoader

    with pytest.warns(DeprecationWarning, match="torch.utils.data.DataLoader"):
        assert pinnse.DataLoader is DataLoader
    assert "DataLoader" not in pinnse.__all__


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        pinnse.nope  # noqa: B018
