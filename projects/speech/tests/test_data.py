import h5py
import numpy as np

from speech.data import _hdf5_to_npz, load_split
from speech.train import bin_trial


def test_hdf5_roundtrip_and_binning(tmp_path):
    rng = np.random.default_rng(0)
    h5 = tmp_path / "x.hdf5"
    feats = [rng.normal(size=(T, 512)).astype(np.float32) for T in (13, 20)]
    with h5py.File(h5, "w") as f:
        for i, x in enumerate(feats):
            g = f.create_group(f"trial_{i:04d}")
            g["input_features"] = x
            g["seq_class_ids"] = np.array([5, 9, 40] + [0] * 7, np.int32)
            g.attrs.update(seq_len=3, sentence_label=f"sentence {i}", block_num=2, trial_num=i, n_time_steps=len(x))
    (tmp_path / "sessions").mkdir()
    n = _hdf5_to_npz(h5, tmp_path / "sessions" / "t15.2000.01.01_train.npz")
    assert n == 2
    trials = load_split(tmp_path, ["t15.2000.01.01"], "train")
    assert [t.sentence for t in trials] == ["sentence 0", "sentence 1"]
    assert trials[1].phones.tolist() == [5, 9, 40]
    assert np.allclose(trials[0].features, feats[0], atol=1e-2)  # float16 storage
    b = bin_trial(trials[0], 3)
    assert b.features.shape == (4, 512)
    assert np.allclose(b.features[0], feats[0][:3].mean(0), atol=1e-2)
