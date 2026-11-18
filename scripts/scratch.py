import cv2
import matplotlib.pyplot as plt
import numpy as np

root = "data/rgbd_dataset_freiburg1_xyz/rgbd_dataset_freiburg1_xyz"
PLAY_VIDEO = True   # 第 2 步的播放；调第 3 步时设为 False 跳过，省得每次看完 798 帧

# ---------------- 第 2 步：读 rgb.txt 并播放 ----------------
timestamps = []
filenames = []
with open(f"{root}/rgb.txt") as f:
    for line in f:
        line = line.strip()                 # 去掉首尾空白和换行符
        if not line or line.startswith("#"):
            continue                        # 空行或注释行,直接跳过
        parts = line.split()                # 按空白拆成 [时间戳, 文件名]
        timestamps.append(float(parts[0])) #时间戳
        filenames.append(parts[1]) #文件名

print(len(timestamps), timestamps[:3], filenames[:3])

if PLAY_VIDEO:
    for name in filenames:
        path = f"{root}/{name}"
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("读图失败:", path)
            break
        cv2.imshow("frame", img)
        if cv2.waitKey(30) == ord("q"):   # 按 q 退出
            break
    cv2.destroyAllWindows()

# ---------------- 第 3 步：读真值并画轨迹 ----------------
# groundtruth.txt 每行: timestamp tx ty tz qx qy qz qw
#   (tx, ty, tz)       相机光心在世界坐标系中的位置，单位米
#   (qx, qy, qz, qw)   相机朝向的四元数（这一步先不用）
# 注意：这是 T_wc（相机 -> 世界）的平移，也就是"相机在哪"，
#       不是我们之后 SLAM 内部用的 T_cw 的平移。两者关系：t_cw = -R_cw @ t_wc
# np.loadtxt 默认把 '#' 开头的行当注释跳过，所以可以直接读
gt = np.loadtxt(f"{root}/groundtruth.txt")
print("真值形状:", gt.shape)          # (N, 8)

t_gt = gt[:, 0]                        # 时间戳
xyz = gt[:, 1:4]                       # 位置，N x 3

print(f"真值时间范围 {t_gt[0]:.2f} ~ {t_gt[-1]:.2f}，共 {t_gt[-1] - t_gt[0]:.1f} 秒，"
      f"平均频率 {len(t_gt) / (t_gt[-1] - t_gt[0]):.2f} Hz")
print(f"图像时间范围 {timestamps[0]:.2f} ~ {timestamps[-1]:.2f}，"
      f"平均频率 {len(timestamps) / (timestamps[-1] - timestamps[0]):.2f} Hz")
print("位置最小值 (m):", np.round(xyz.min(axis=0), 3))
print("位置最大值 (m):", np.round(xyz.max(axis=0), 3))
print("各轴运动范围 (m):", np.round(np.ptp(xyz, axis=0), 3))

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]   # 让图里能显示中文
plt.rcParams["axes.unicode_minus"] = False                          # 负号正常显示
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 俯视/侧视两张二维投影，axis("equal") 保证 x、y 比例一致，不会把轨迹拉变形
axes[0].plot(xyz[:, 0], xyz[:, 1], lw=0.8)
axes[0].plot(*xyz[0, :2], "go", label="起点")
axes[0].set(xlabel="x (m)", ylabel="y (m)", title="x-y 投影")
axes[0].axis("equal")
axes[0].legend()

axes[1].plot(xyz[:, 0], xyz[:, 2], lw=0.8)
axes[1].plot(xyz[0, 0], xyz[0, 2], "go", label="起点")
axes[1].set(xlabel="x (m)", ylabel="z (m)", title="x-z 投影")
axes[1].axis("equal")
axes[1].legend()

# 三个坐标随时间变化：fr1/xyz 是先后沿三个方向来回晃，这张图最容易看出来
t_rel = t_gt - t_gt[0]
for k, name in enumerate("xyz"):
    axes[2].plot(t_rel, xyz[:, k] - xyz[0, k], label=name)
# 图像覆盖的时间段：真值比图像早开始几秒
axes[2].axvspan(timestamps[0] - t_gt[0], timestamps[-1] - t_gt[0], color="gray", alpha=0.15, label="有图像的时段")
axes[2].set(xlabel="时间 (s)", ylabel="相对起点的位移 (m)", title="位置随时间变化")
axes[2].legend()

plt.tight_layout()
plt.savefig("results/stage0_gt_trajectory.png", dpi=120)
plt.show()
# ---------------- 第 4 步：时间戳关联 ----------------
img_t = np.array(timestamps)       # 列表转数组，才能整体做减法

matched_idx = []                   # 每张图对应的真值行号
time_diffs = []                    # 每张图对应的时间差

for t in img_t:
    diff = np.abs(t_gt - t)        # 先算：这张图和所有真值的时间差
    j = np.argmin(diff)            # 再找：最接近的那一行
    matched_idx.append(j)          # 最后存
    time_diffs.append(diff[j])

matched_idx = np.array(matched_idx)
time_diffs = np.array(time_diffs)
# ---------------- 第 4c 步：检查关联结果 ----------------
# 写完一段代码，要用数字检查结果是否合理，而不是只看它有没有报错
print("最大时间差:", time_diffs.max())
print("平均时间差:", time_diffs.mean())
print("超过 0.02 秒的图像数:", np.sum(time_diffs > 0.02))
# 如果 diff/j 写在循环外面，所有图会对到同一行，这里会打印 1
print("对到了几个不同的真值行:", len(np.unique(matched_idx)))

gt_for_images = gt[matched_idx]    # 每张图对应的那一行真值，形状应该是 (798, 8)
print(gt_for_images.shape)

# 第 4 步的成果：第 i 张图拍摄时，相机在 gt_for_images[i] 那个位置。
# 这份草稿的逻辑已经整理进 orbslam/dataset.py，之后的代码都调用那里的 TUMDataset。
