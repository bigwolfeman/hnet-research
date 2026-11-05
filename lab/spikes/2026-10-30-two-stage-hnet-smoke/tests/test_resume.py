"""train.py resume files: durable write, torn-file fallback, config guard."""
import pytest
import torch

from train import config_diff, load_resume, save_resume

CFG = {"seed": 0, "train": {"lr": 1e-3, "lr_mult": [1.5, 1.0]}}


def test_config_diff_paths():
    assert config_diff(CFG, CFG) == []
    other = {"seed": 0, "train": {"lr": 2e-3, "lr_mult": [1.5, 1.0], "extra": 1}}
    assert config_diff(CFG, other) == ["train.extra", "train.lr"]


def test_save_load_round_trip_and_rotation(tmp_path):
    assert load_resume(tmp_path, CFG) is None
    save_resume(tmp_path, dict(step=8, cfg=CFG, w=torch.arange(3)))
    save_resume(tmp_path, dict(step=16, cfg=CFG, w=torch.arange(4)))
    ck = load_resume(tmp_path, CFG)
    assert ck["step"] == 16 and torch.equal(ck["w"], torch.arange(4))
    assert torch.load(tmp_path / "resume_prev.pt", weights_only=False)["step"] == 8
    assert not (tmp_path / "resume.pt.tmp").exists()


def test_torn_newest_falls_back_to_previous(tmp_path):
    save_resume(tmp_path, dict(step=8, cfg=CFG))
    save_resume(tmp_path, dict(step=16, cfg=CFG))
    data = (tmp_path / "resume.pt").read_bytes()
    (tmp_path / "resume.pt").write_bytes(data[: len(data) // 2])   # power loss in the middle of the write
    assert load_resume(tmp_path, CFG)["step"] == 8
    (tmp_path / "resume_prev.pt").write_bytes(b"\0" * 64)
    with pytest.raises(RuntimeError, match="none is readable"):
        load_resume(tmp_path, CFG)


def test_config_mismatch_is_an_error(tmp_path):
    save_resume(tmp_path, dict(step=8, cfg=CFG))
    with pytest.raises(AssertionError, match="train.lr"):
        load_resume(tmp_path, {"seed": 0, "train": {"lr": 2e-3, "lr_mult": [1.5, 1.0]}})
