# 阶段 0 + 阶段 1 任务单

约定：代码你来写，写完一个文件就可以让我审查。下面的函数签名是建议，不是强制——你觉得别的组织方式更顺手可以改，但**输入输出的含义和单位**请保持一致，后面阶段会依赖它们。

运行方式统一为 `uv run python scripts/xxx.py`、`uv run pytest`。

---

## 阶段 0　环境与数据（✅ 已完成：orbslam/dataset.py、scripts/stage0_view.py、tests/test_dataset.py）

### 0.1 环境（已完成）

```bash
uv run python scripts/check_env.py
```

最后一行打印 `环境 OK` 即可。环境固定为 Python 3.11 + numpy<2 + g2o-python 0.0.12，原因写在 `pyproject.toml` 的注释里，不要随意升级。

### 0.2 下载数据

从 TUM RGB-D 数据集下载页 <https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download> 下载 `freiburg1_xyz`（tgz，约 0.5 GB），解压到：

```
data/rgbd_dataset_freiburg1_xyz/
  rgb/            # 时间戳命名的 png
  rgb.txt         # 每行: timestamp  rgb/xxx.png
  groundtruth.txt # 每行: timestamp tx ty tz qx qy qz qw
```

我们只用 RGB 图像和 groundtruth，depth 用不到。

### 0.3 实现 `orbslam/dataset.py`

```python
@dataclass
class Camera:
    K: np.ndarray        # 3x3，去畸变之后使用的内参
    dist: np.ndarray     # 原始畸变系数 (k1, k2, p1, p2, k3)
    K_raw: np.ndarray    # 3x3，原始内参
    width: int
    height: int

class TUMDataset:
    def __init__(self, root: str): ...
    def __len__(self) -> int: ...
    def __getitem__(self, i) -> tuple[float, np.ndarray]:
        """返回 (时间戳, 灰度图 uint8 HxW)。"""
    camera: Camera
    def groundtruth(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 (frame_idx, gt)：有真值的图像下标，以及对应的 Kx8 真值
        [t, tx, ty, tz, qx, qy, qz, qw]，按最近时间戳关联。"""
```

要点：

- **内参**（来自 TUM 官方标定；请到数据集网站的 "Intrinsic Camera Calibration" 页面自己核对一遍）：

  | 序列 | fx | fy | cx | cy | k1 | k2 | p1 | p2 | k3 |
  |---|---|---|---|---|---|---|---|---|---|
  | fr1 | 517.3 | 516.5 | 318.6 | 255.3 | 0.2624 | -0.9531 | -0.0054 | 0.0026 | 1.1633 |
  | fr2 | 520.9 | 521.0 | 325.1 | 249.7 | 0.2312 | -0.7849 | -0.0033 | -0.0001 | 0.9172 |
  | fr3 | 535.4 | 539.2 | 320.1 | 247.6 | 0 | 0 | 0 | 0 | 0 |

- **去畸变策略**：不要整图去畸变，而是提取特征后用 `cv2.undistortPoints(pts, K_raw, dist, P=K)` 只对关键点坐标去畸变（ORB-SLAM 也是这么做的）。
- **`K` 取 fx = fy**：g2o 的 `CameraParameters` 只支持单一焦距（见 `scripts/check_env.py` 顶部说明）。把 `K` 的焦距设成 `(fx + fy) / 2`，主点不变，作为 `undistortPoints` 的 `P` 传入，这样之后所有模块用的都是这个"理想针孔相机"。
- 时间戳关联：groundtruth 是 100 Hz，图像是 30 Hz，对每张图找时间差最小的真值，差值超过 0.02 s 的丢弃。

### 0.4 验收

写 `scripts/stage0_view.py`：

1. 逐帧 `cv2.imshow` 播放 fr1/xyz，窗口标题或左上角显示帧号和时间戳。
2. 用 matplotlib 画出真值轨迹的 x-y、x-z 投影。fr1/xyz 的相机主要沿三个轴来回平移，轨迹范围大约几十厘米，看到的应该是沿坐标轴方向的来回线段。
3. 命令行验证 evo 可用：

```bash
uv run evo_traj tum data/rgbd_dataset_freiburg1_xyz/groundtruth.txt --plot
```

---

## 阶段 1　ORB 特征与匹配

对应论文：III-A（特征选择）、V-A（ORB 提取）、IV 第 1 步（初始化时的匹配）。

### 1.1 实现 `orbslam/features.py` —— 提取

```python
N_LEVELS = 8
SCALE_FACTOR = 1.2

class ORBExtractor:
    def __init__(self, n_features=1000, scale_factor=1.2, n_levels=8): ...
    scale_factors: np.ndarray   # [1.2**i for i in range(8)]
    level_sigma2: np.ndarray    # scale_factors**2，后面 BA 里作为观测协方差用

    def extract(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """返回
        pts      Nx2 float32  原图像素坐标（未去畸变）
        octaves  N   int      所在金字塔层
        angles   N   float32  主方向（度）
        desc     Nx32 uint8   描述子
        """
```

分两步做：

- **版本 A（先做这个）**：`cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8, fastThreshold=7)` 多提一些点，然后把图像划成网格（比如 64×48 像素一格 → 10×10 格），每格按 `response` 从高到低保留，使总数接近 1000 且尽量均匀。
- **版本 B（可选，阶段 5 之后有余力再做）**：论文原方法——自己建金字塔，每层划网格，每格先用 FAST 阈值 20 检测，检测不到再降到 7，然后按 response 筛选，最后统一计算方向和描述子。源码见 ORB_SLAM2 的 `ORBextractor.cc`。

需要想清楚并在注释里回答的问题：**为什么要均匀化？**（提示：考虑位姿估计的几何条件，以及纹理集中在画面一角时会发生什么。）

### 1.2 实现 `orbslam/features.py` —— 匹配

```python
def hamming_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: Mx32, b: Nx32 uint8 -> MxN 距离矩阵。自己用 numpy 实现（查表或 np.unpackbits）。"""

def match_brute_force(desc1, desc2, max_dist=50, ratio=0.9) -> np.ndarray:
    """返回 Kx2 的索引对 (i1, i2)。
    规则：最近邻距离 <= max_dist；最近邻 < ratio * 次近邻；一对一（互为最近邻或冲突时保留距离小者）。"""

def filter_by_rotation(matches, angles1, angles2, n_bins=30, keep_top=3) -> np.ndarray:
    """旋转一致性检验：对每对匹配计算角度差，放进 30 个 bin 的直方图，只保留数量最多的 3 个 bin 里的匹配。"""

def match_in_window(pts1, desc1, octaves1, pts2, desc2, octaves2, radius=100, ...) -> np.ndarray:
    """初始化用的匹配：对第 1 帧每个特征，只在第 2 帧相同位置附近 radius 像素窗口内搜索。"""
```

说明：

- 阈值 50、比值 0.9、30 个 bin 取前 3 都来自 ORB_SLAM2 源码 `ORBmatcher.cc`（`TH_LOW`、`mfNNratio`、`HISTO_LENGTH`），论文正文没有写。
- `match_in_window` 需要按位置快速查找特征：把特征按网格分桶（`Frame::GetFeaturesInArea` 的做法），这个结构阶段 4 的投影匹配还会用到，值得现在写好。
- `hamming_distance` 请自己写一遍再去用 `cv2.BFMatcher(cv2.NORM_HAMMING)` 对比结果和速度，之后两者用哪个都行。

### 1.3 单元测试 `tests/test_features.py`

- `hamming_distance`：与逐比特暴力计算的结果一致；自身距离为 0；对称。
- `filter_by_rotation`：构造 100 对角度差集中在 20° 附近的匹配 + 20 对随机角度差的匹配，断言随机的大部分被剔除、集中的全部保留。
- 把一张图旋转 30°（`cv2.warpAffine`）后与原图匹配，匹配点对应的坐标满足该旋转变换的比例应当很高。

### 1.4 验收 `scripts/stage1_matching.py`

取 fr1/xyz 的第 0 帧和第 10 帧：

1. 分别画出均匀化前后的关键点分布，统计每个网格的特征数（打印最小值、最大值、空格子数）。
2. 用 `cv2.drawMatches` 画出三种结果：只做暴力匹配 / 加比值测试 / 再加旋转一致性检验，对比误匹配减少的情况。
3. 量化检查：用 `cv2.findFundamentalMat(..., cv2.FM_RANSAC, 1.0)` 对最终匹配估计 F，打印匹配总数和内点率。合理的结果是有几百对匹配且大部分是内点；把你得到的具体数字记下来，后面改代码时可以作为回归基准。
4. 把帧间隔换成 1、5、30 各跑一遍，观察匹配数和内点率的变化——这会让你直观理解论文第 IV 章为什么要"等到视差足够"才初始化。

### 完成标志

- `uv run pytest` 全部通过；
- 两个验收脚本能跑，并且你能解释每一步过滤掉的是哪一类误匹配。

完成后把代码给我审查，我再给阶段 2（地图初始化）的任务单。
