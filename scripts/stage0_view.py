"""阶段 0 验收脚本：播放序列 + 画关联后的真值轨迹 + 看一眼畸变有多大。

运行：
    uv run python scripts/stage0_view.py                 # 默认 fr1/xyz
    uv run python scripts/stage0_view.py --no-video      # 跳过播放
    uv run python scripts/stage0_view.py data/rgbd_dataset_freiburg2_desk

播放时：空格 暂停/继续，q 退出。
图保存在 results/stage0_*.png。
"""
import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # 让脚本能 import orbslam
from orbslam.dataset import TUMDataset

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]   # 让图里能显示中文
plt.rcParams["axes.unicode_minus"] = False                          # 负号正常显示


def play(ds: TUMDataset):
    """逐帧播放，左上角显示帧号和时间戳。"""
    paused = False
    i = 0
    while i < len(ds):
        t, img = ds[i]
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.putText(vis, f"#{i}  t={t:.3f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("TUM sequence", vis)
        key = cv2.waitKey(0 if paused else 30)
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
        if not paused:
            i += 1
    cv2.destroyAllWindows()


def plot_groundtruth(ds: TUMDataset, out_path: str):
    """和 scratch.py 第 3 步一样的三张图，但用的是"关联到图像上"的真值。"""
    frame_idx, gt = ds.groundtruth()
    xyz = gt[:, 1:4]
    print(f"图像 {len(ds)} 张，其中 {len(frame_idx)} 张有对应真值"
          f"（丢弃 {len(ds) - len(frame_idx)} 张：真值在这些时刻有缺口）")
    dt = np.abs(gt[:, 0] - ds.timestamps[frame_idx])
    print(f"保留的图像与真值的时间差：最大 {dt.max() * 1000:.1f} ms，平均 {dt.mean() * 1000:.1f} ms")
    print("各轴运动范围 (m):", np.round(np.ptp(xyz, axis=0), 3))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (a, b) in zip(axes[:2], [(0, 1), (0, 2)]):
        ax.plot(xyz[:, a], xyz[:, b], lw=0.8)
        ax.plot(xyz[0, a], xyz[0, b], "go", label="起点")
        ax.set(xlabel=f"{'xyz'[a]} (m)", ylabel=f"{'xyz'[b]} (m)", title=f"{'xyz'[a]}-{'xyz'[b]} 投影")
        ax.axis("equal")
        ax.legend()

    t_rel = ds.timestamps[frame_idx] - ds.timestamps[0]
    for k, name in enumerate("xyz"):
        axes[2].plot(t_rel, xyz[:, k] - xyz[0, k], label=name)
    axes[2].set(xlabel="时间 (s)", ylabel="相对起点的位移 (m)", title="位置随时间变化（只含有图像的时段）")
    axes[2].legend()

    fig.suptitle(os.path.basename(ds.root))
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print("已保存", out_path)


def plot_distortion(ds: TUMDataset, out_path: str):
    """畸变有多大：在原图上取一个网格，看每个点去畸变后挪了多少像素。

    这决定了后面"不去畸变会怎样"：图像边缘偏差十几个像素时，
    三角化和位姿估计都会明显变差，所以关键点一定要先去畸变。
    """
    cam = ds.camera
    u, v = np.meshgrid(np.linspace(0, cam.width - 1, 17), np.linspace(0, cam.height - 1, 13))
    pts = np.stack([u.ravel(), v.ravel()], axis=1)
    pts_u = cam.undistort_points(pts)
    shift = np.linalg.norm(pts_u - pts, axis=1)
    print(f"去畸变位移：图像中心附近 {shift[np.argmin(np.linalg.norm(pts - [cam.cx, cam.cy], axis=1))]:.2f} px，"
          f"最大 {shift.max():.1f} px（在角落）")

    _, img = ds[0]
    undist_img = cv2.undistort(img, cam.K_raw, cam.dist, None, cam.K)  # 整图去畸变，只用来看效果

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("原图（带畸变）")
    axes[1].imshow(undist_img, cmap="gray")
    axes[1].set_title("整图去畸变（仅用于对比，SLAM 中只对关键点去畸变）")
    axes[2].quiver(pts[:, 0], pts[:, 1], pts_u[:, 0] - pts[:, 0], pts_u[:, 1] - pts[:, 1],
                   shift, angles="xy", scale_units="xy", scale=0.1, cmap="viridis")
    axes[2].set(xlim=(-20, cam.width + 20), ylim=(cam.height + 20, -20), aspect="equal",
                title="去畸变位移场（箭头放大 10 倍）")
    for ax in axes[:2]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    print("已保存", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default="data/rgbd_dataset_freiburg1_xyz")
    parser.add_argument("--no-video", action="store_true", help="跳过逐帧播放")
    args = parser.parse_args()

    ds = TUMDataset(args.root)
    print(f"序列 {ds.root}：{len(ds)} 张图，时长 {ds.timestamps[-1] - ds.timestamps[0]:.1f} s")
    print("去畸变后使用的内参 K =\n", np.round(ds.camera.K, 2))

    if not args.no_video:
        play(ds)

    os.makedirs("results", exist_ok=True)
    plot_groundtruth(ds, "results/stage0_gt_trajectory.png")
    plot_distortion(ds, "results/stage0_distortion.png")
    plt.show()


if __name__ == "__main__":
    main()
