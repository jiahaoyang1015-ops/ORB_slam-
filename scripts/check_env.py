"""环境自检：依赖能否导入 + g2o 能否跑通一个最小的两视图 BA。

运行： uv run python scripts/check_env.py
预期： chi2 从上万降到几十，最后打印 "环境 OK"。

这个脚本同时是 g2o-python 0.0.12 的用法示例（该绑定几乎没有文档），
阶段 5 写局部 BA 时可以回来参考。两个实测得到的注意点：

1. 投影边要用 EdgeProjectXYZ2UV + CameraParameters。
   EdgeSE3ProjectXYZ（ORB-SLAM2 C++ 里用的那个）在这个绑定里没有暴露 fx/fy/cx/cy，
   内参无法设置，优化结果是错的。
2. CameraParameters 只有一个焦距（假设 fx == fy）。
   TUM 的 fx、fy 相差不到 1 像素，去畸变时把新内参设成 fx = fy 即可（见阶段 0 任务单）。
"""
import cv2
import evo  # noqa: F401  只检查能否导入
import g2o
import matplotlib  # noqa: F401
import numpy as np
import scipy

print(f"numpy {np.__version__} | opencv {cv2.__version__} | scipy {scipy.__version__}")
assert int(np.__version__.split(".")[0]) < 2, "g2o-python 0.0.12 需要 numpy<2"

rng = np.random.default_rng(0)
f, cx, cy = 500.0, 320.0, 240.0

# 真值：60 个三维点，两个相机位姿 T_cw（世界 -> 相机）
points = rng.uniform([-1, -1, 3], [1, 1, 6], size=(60, 3))
poses = [
    g2o.SE3Quat(np.eye(3), np.array([0.0, 0.0, 0.0])),
    g2o.SE3Quat(np.eye(3), np.array([-0.3, 0.0, 0.0])),
]

opt = g2o.SparseOptimizer()
opt.set_algorithm(
    g2o.OptimizationAlgorithmLevenberg(g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3()))
)

cam = g2o.CameraParameters(f, np.array([cx, cy]), 0.0)
cam.set_id(0)
opt.add_parameter(cam)

# 位姿顶点：第 0 帧固定（消除规范自由度），第 1 帧加噪声作为初值
for i, T in enumerate(poses):
    v = g2o.VertexSE3Expmap()
    v.set_id(i)
    noise = 0.05 * rng.normal(size=3) if i > 0 else np.zeros(3)
    v.set_estimate(g2o.SE3Quat(np.eye(3), T.translation() + noise))
    v.set_fixed(i == 0)
    opt.add_vertex(v)

# 地图点顶点 + 重投影误差边
for j, p in enumerate(points):
    v = g2o.VertexPointXYZ()
    v.set_id(100 + j)
    v.set_marginalized(True)  # Schur 消元
    v.set_estimate(p + 0.05 * rng.normal(size=3))
    opt.add_vertex(v)
    for i, T in enumerate(poses):
        pc = T.map(p)
        uv = np.array([f * pc[0] / pc[2] + cx, f * pc[1] / pc[2] + cy])
        e = g2o.EdgeProjectXYZ2UV()
        e.set_vertex(0, v)              # 顶点 0：地图点
        e.set_vertex(1, opt.vertex(i))  # 顶点 1：位姿
        e.set_measurement(uv + 0.5 * rng.normal(size=2))
        e.set_information(np.eye(2))
        e.set_robust_kernel(g2o.RobustKernelHuber(np.sqrt(5.991)))
        e.set_parameter_id(0, 0)        # 绑定上面 id=0 的相机参数
        opt.add_edge(e)

opt.initialize_optimization()
opt.compute_active_errors()
chi2_before = opt.chi2()
opt.optimize(10)
chi2_after = opt.chi2()

print(f"chi2: {chi2_before:.1f} -> {chi2_after:.1f}")
print("第 1 帧平移估计:", np.round(opt.vertex(1).estimate().translation(), 3), "(真值方向 [-1, 0, 0])")
assert chi2_after < 0.01 * chi2_before, "BA 没有收敛"
print("环境 OK")
