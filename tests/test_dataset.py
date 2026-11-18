"""orbslam/dataset.py 的单元测试。运行：uv run pytest"""
import os

import cv2
import numpy as np
import pytest

from orbslam.dataset import TUM_CALIB, Camera, TUMDataset, associate, read_file_list

DATA = "data/rgbd_dataset_freiburg1_xyz"


def test_read_file_list(tmp_path):
    p = tmp_path / "rgb.txt"
    p.write_text("# color images\n# timestamp filename\n\n"
                 "1.5 rgb/1.5.png\n2.0 rgb/2.0.png\n")
    t, entries = read_file_list(str(p))
    assert np.allclose(t, [1.5, 2.0])
    assert entries == [["rgb/1.5.png"], ["rgb/2.0.png"]]


def test_associate_matches_brute_force_loop():
    """searchsorted 版本必须和 scratch.py 第 4 步的 for 循环结果一致。"""
    rng = np.random.default_rng(0)
    ref = np.sort(rng.uniform(0, 10, 1000))
    query = rng.uniform(-1, 11, 300)          # 故意包含超出 ref 范围的时间
    idx, valid = associate(query, ref, max_diff=0.02)
    for q, j, ok in zip(query, idx, valid):
        diff = np.abs(ref - q)
        assert j == np.argmin(diff)
        assert ok == (diff.min() <= 0.02)


def test_undistort_identity_without_distortion():
    """没有畸变、且 fx == fy 时，去畸变不应改变坐标。"""
    cam = Camera.from_calib(500, 500, 320, 240, [0, 0, 0, 0, 0])
    pts = np.array([[0, 0], [320, 240], [639, 479]], dtype=float)
    assert np.allclose(cam.undistort_points(pts), pts, atol=1e-6)


def test_undistort_inverts_distortion():
    """先用畸变模型把理想点"弄歪"，再去畸变，应该回到理想点。"""
    cam = Camera.from_calib(**TUM_CALIB["freiburg1"])
    rng = np.random.default_rng(1)
    xy = rng.uniform([-0.5, -0.4], [0.5, 0.4], size=(50, 2))       # 理想归一化坐标
    obj = np.hstack([xy, np.ones((50, 1))])                         # 深度 1 的三维点
    distorted, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), cam.K_raw, cam.dist)
    ideal = (cam.K @ obj.T).T[:, :2]                                # 用 K 投影的理想像素
    assert np.allclose(cam.undistort_points(distorted.reshape(-1, 2)), ideal, atol=0.05)


@pytest.mark.skipif(not os.path.isdir(DATA), reason="没有下载 fr1/xyz")
def test_fr1_xyz_dataset():
    ds = TUMDataset(DATA)
    assert len(ds) == 798
    t, img = ds[0]
    assert img.shape == (480, 640) and img.dtype == np.uint8
    assert t == ds.timestamps[0]

    frame_idx, gt = ds.groundtruth()
    assert gt.shape == (len(frame_idx), 8)
    assert len(frame_idx) > 0.99 * len(ds)
    assert np.abs(gt[:, 0] - ds.timestamps[frame_idx]).max() <= 0.02
    q_norm = np.linalg.norm(gt[:, 4:8], axis=1)
    assert np.allclose(q_norm, 1, atol=1e-3)    # 四元数应为单位长度
