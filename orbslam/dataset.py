"""TUM RGB-D 数据集读取（只用 RGB 图像 + 真值轨迹）。

用法：
    from orbslam.dataset import TUMDataset
    ds = TUMDataset("data/rgbd_dataset_freiburg1_xyz")
    t, gray = ds[0]                          # 时间戳, 灰度图 uint8 HxW
    pts_u = ds.camera.undistort_points(pts)  # 原图像素坐标 -> 去畸变后的像素坐标
    frame_idx, gt = ds.groundtruth()         # 与图像关联好的真值

坐标约定（全项目统一）：
    真值文件里的 (tx, ty, tz, qx, qy, qz, qw) 是 T_wc，即"相机 -> 世界"，
    平移部分就是相机光心在世界坐标系中的位置。
    SLAM 内部统一用 T_cw（"世界 -> 相机"），两者互为逆：T_cw = T_wc^-1。

逻辑来源：scripts/scratch.py 第 2～4 步的草稿。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 相机标定参数（TUM 官方页面 cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats）
#   fx, fy, cx, cy : 针孔内参（像素）
#   dist           : OpenCV 畸变系数顺序 (k1, k2, p1, p2, k3)，即页面上的 d0..d4
# fr3 的图像官方已经去过畸变，所以畸变系数全为 0。
# ---------------------------------------------------------------------------
TUM_CALIB = {
    "freiburg1": dict(fx=517.3, fy=516.5, cx=318.6, cy=255.3,
                      dist=[0.2624, -0.9531, -0.0054, 0.0026, 1.1633]),
    "freiburg2": dict(fx=520.9, fy=521.0, cx=325.1, cy=249.7,
                      dist=[0.2312, -0.7849, -0.0033, -0.0001, 0.9172]),
    "freiburg3": dict(fx=535.4, fy=539.2, cx=320.1, cy=247.6,
                      dist=[0.0, 0.0, 0.0, 0.0, 0.0]),
}

MAX_TIME_DIFF = 0.02   # 图像与真值时间戳相差超过 20 ms 就认为没有对应的真值


@dataclass
class Camera:
    """针孔相机模型。

    去畸变策略（与 ORB-SLAM 相同）：不对整张图去畸变，而是在原图上提特征，
    之后只对关键点坐标调用 undistort_points。这样既省时间，又不会因为图像插值
    损坏 FAST 角点。

    K 与 K_raw 的区别：
        K_raw  原始标定内参，只在去畸变时使用
        K      去畸变后使用的"理想针孔相机"内参，之后所有几何模块（三角化、
               PnP、BA ...）都只用 K。这里取 fx = fy = (fx_raw + fy_raw) / 2，
               因为 g2o-python 的 CameraParameters 只支持单一焦距
               （见 scripts/check_env.py 的说明）。
    """
    K: np.ndarray        # 3x3，去畸变之后使用的内参
    dist: np.ndarray     # 原始畸变系数 (k1, k2, p1, p2, k3)
    K_raw: np.ndarray    # 3x3，原始内参
    width: int
    height: int

    @classmethod
    def from_calib(cls, fx, fy, cx, cy, dist, width=640, height=480) -> "Camera":
        K_raw = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float64)
        f = 0.5 * (fx + fy)
        K = np.array([[f, 0, cx],
                      [0, f, cy],
                      [0, 0, 1]], dtype=np.float64)
        return cls(K=K, dist=np.asarray(dist, dtype=np.float64), K_raw=K_raw,
                   width=width, height=height)

    # 方便后面写公式时直接用 cam.fx 之类
    @property
    def fx(self) -> float:
        return self.K[0, 0]

    @property
    def fy(self) -> float:
        return self.K[1, 1]

    @property
    def cx(self) -> float:
        return self.K[0, 2]

    @property
    def cy(self) -> float:
        return self.K[1, 2]

    def undistort_points(self, pts: np.ndarray) -> np.ndarray:
        """原图像素坐标 (Nx2) -> 去畸变后、在内参 K 下的像素坐标 (Nx2)。

        cv2.undistortPoints 内部做的事：
            1. 用 K_raw 把像素坐标变成（带畸变的）归一化坐标
            2. 迭代求解畸变模型的逆，得到无畸变的归一化坐标 (x, y)
            3. 因为传了 P=K，再用 K 投影回像素坐标；不传 P 则返回归一化坐标
        """
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        if len(pts) == 0:
            return np.zeros((0, 2))
        out = cv2.undistortPoints(pts, self.K_raw, self.dist, P=self.K)
        return out.reshape(-1, 2)


def read_file_list(path: str) -> tuple[np.ndarray, list[list[str]]]:
    """读 TUM 格式的列表文件（rgb.txt / groundtruth.txt ...）。

    每行: 时间戳 其余字段...；'#' 开头的行和空行跳过。
    返回 (时间戳数组, 每行除时间戳外的字段列表)。
    """
    timestamps = []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()                 # 去掉首尾空白和换行符
            if not line or line.startswith("#"):
                continue                        # 空行或注释行，直接跳过
            parts = line.split()                # 按空白拆成 [时间戳, ...]
            timestamps.append(float(parts[0]))
            entries.append(parts[1:])
    return np.array(timestamps), entries


def associate(query_t: np.ndarray, ref_t: np.ndarray,
              max_diff: float = MAX_TIME_DIFF) -> tuple[np.ndarray, np.ndarray]:
    """对 query_t 中每个时间戳，在 ref_t 中找时间最接近的一个。

    返回
        idx    长度 len(query_t)，最接近的 ref 下标
        valid  布尔数组，时间差 <= max_diff 的为 True

    做法和 scratch.py 第 4 步的 for 循环结果完全一样（tests 里有对比），
    只是换成 np.searchsorted，不用对每张图都扫一遍全部真值：
        searchsorted 要求 ref_t 有序，返回"把 t 插进去后仍保持有序"的位置 k，
        即 ref_t[k-1] < t <= ref_t[k]。最近的那个只可能是 k-1 或 k，比一下即可。
    """
    query_t = np.asarray(query_t, dtype=np.float64)
    ref_t = np.asarray(ref_t, dtype=np.float64)
    assert np.all(np.diff(ref_t) >= 0), "ref_t 必须按时间排序"

    k = np.searchsorted(ref_t, query_t)
    left = np.clip(k - 1, 0, len(ref_t) - 1)    # 左边的候选
    right = np.clip(k, 0, len(ref_t) - 1)       # 右边的候选（clip 处理越界）
    use_right = np.abs(ref_t[right] - query_t) < np.abs(ref_t[left] - query_t)
    idx = np.where(use_right, right, left)
    valid = np.abs(ref_t[idx] - query_t) <= max_diff
    return idx, valid


def _find_sequence_root(root: str) -> str:
    """有的解压工具会多套一层同名文件夹，这里自动往下找到含 rgb.txt 的那一层。"""
    if os.path.isfile(os.path.join(root, "rgb.txt")):
        return root
    nested = os.path.join(root, os.path.basename(os.path.normpath(root)))
    if os.path.isfile(os.path.join(nested, "rgb.txt")):
        return nested
    raise FileNotFoundError(f"在 {root} 及其同名子目录中找不到 rgb.txt")


def _guess_calib_name(root: str) -> str:
    name = os.path.basename(os.path.normpath(root))
    for key in TUM_CALIB:
        if key in name:
            return key
    raise ValueError(f"无法从目录名 {name} 判断是 freiburg1/2/3，请显式传入 calib=")


class TUMDataset:
    """一个 TUM RGB-D 序列。

    ds[i]            -> (时间戳, 灰度图)
    len(ds)          -> 图像数
    ds.timestamps    -> 所有图像时间戳 (N,)
    ds.camera        -> Camera
    ds.groundtruth() -> (frame_idx, gt) 与图像关联好的真值
    """

    def __init__(self, root: str, calib: str | None = None):
        self.root = _find_sequence_root(root)
        calib = calib or _guess_calib_name(self.root)
        self.camera = Camera.from_calib(**TUM_CALIB[calib])

        # 图像列表：rgb.txt 每行 "时间戳 rgb/xxx.png"
        self.timestamps, entries = read_file_list(os.path.join(self.root, "rgb.txt"))
        self.image_paths = [os.path.join(self.root, e[0]) for e in entries]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, i: int) -> tuple[float, np.ndarray]:
        """返回 (时间戳, 灰度图 uint8 HxW)。"""
        path = self.image_paths[i]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"读图失败: {path}")
        return float(self.timestamps[i]), img

    def raw_groundtruth(self) -> np.ndarray:
        """未关联的完整真值 Mx8 [t, tx, ty, tz, qx, qy, qz, qw]，100 Hz。"""
        # np.loadtxt 默认把 '#' 开头的行当注释跳过
        return np.loadtxt(os.path.join(self.root, "groundtruth.txt"))

    def groundtruth(self, max_diff: float = MAX_TIME_DIFF) -> tuple[np.ndarray, np.ndarray]:
        """把真值按最近时间戳关联到图像上。

        返回
            frame_idx  (K,)   有真值的图像下标；时间差超过 max_diff 的图像被丢弃
                              （真值在这些时刻有缺口，用不准的位置去评估会引入误差）
            gt         (K, 8) 对应的真值 [t, tx, ty, tz, qx, qy, qz, qw]，
                              第 0 列保留真值自己的时间戳
        第 k 张有效图像 ds[frame_idx[k]] 拍摄时，相机位姿是 gt[k]。
        """
        raw = self.raw_groundtruth()
        idx, valid = associate(self.timestamps, raw[:, 0], max_diff)
        frame_idx = np.nonzero(valid)[0]
        return frame_idx, raw[idx[valid]]
