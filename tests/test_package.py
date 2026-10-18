import pinnse


def test_public_api_is_exported():
    for name in pinnse.__all__:
        assert hasattr(pinnse, name), name


def test_dataloader_is_torch_dataloader():
    from torch.utils.data import DataLoader

    assert pinnse.DataLoader is DataLoader
