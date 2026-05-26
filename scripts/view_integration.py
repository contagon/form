from typing import cast
from evalio import datasets as ds, types as ty
from form import FORMInertial, OxfordSpiresCustom
from form.multi_campus import MultiCampusCustom
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# params
num_lidar = 500
num_after = 50

dataset = OxfordSpiresCustom.blenheim_palace_02
# dataset = ds.OxfordSpires.blenheim_palace_02
dataset = MultiCampusCustom.tuhh_night_09

# setup
exp = ty.Experiment.from_pl_ds(FORMInertial, dataset)
exp.pipeline_params["planar_sigma"] = 0.01
exp.pipeline_params["prior_scale"] = 0.1
exp.pipeline_params["point_feats_per_sector"] = 0
out = exp.setup()
if isinstance(out, Exception):
    raise out
pipe, data = out
pipe = cast(FORMInertial, pipe)

# run
imu_estimates: list[ty.SE3] = []
bias_estimates = []
bias_stamps = []
loop = tqdm(total=num_lidar + num_after)
for mm in data:
    if isinstance(mm, ty.LidarMeasurement):
        if loop.n < num_lidar:
            pipe.add_lidar(mm)
        loop.update(1)
        # print("lidar", mm.stamp + data.lidar_params().delta_time())
    if isinstance(mm, ty.ImuMeasurement):
        pipe.add_imu(mm)
        if loop.n >= 50:
            imu_estimates.append(pipe.get_last_integrated_imu())

        # print("imu", mm.stamp)
    if loop.n >= num_lidar + num_after:
        break

    if loop.n > 50:
        bias_estimates.append(pipe.current_imu_bias())
        bias_stamps.append(mm.stamp.to_sec())

loop.close()

# plot bias estimates
bias_estimates = np.array(bias_estimates)
bias_stamps = np.array(bias_stamps) - bias_stamps[0]
fig, ax = plt.subplots(2, 3, figsize=(9, 6), layout="constrained", sharex=True)

labels = ["bax", "bay", "baz", "bgx", "bgy", "bgz"]
for i in range(2):
    for j in range(3):
        ax[i, j].plot(bias_stamps, bias_estimates[:, 3 * i + j], "bo-")
        ax[i, j].set_title(labels[3 * i + j])
        ax[i, j].grid(True)

plt.savefig("scripts/view_biases.png")
# plt.show()

# # plot lidar poses
lidar_estimates = [p for s, p in pipe.saved_estimates()]
lidar_trans = np.array([p.trans for p in lidar_estimates])
imu_trans = np.array([p.trans for p in imu_estimates])

print(f"Num lidar poses; {lidar_trans.shape}")
print(f"Num imu poses; {imu_trans.shape}")

# Truncate to num_after before and after
num_imu_per_lidar = int(400 / dataset.lidar_params().rate)
imu_trans = imu_trans[-(num_after * 2 * num_imu_per_lidar) :]
lidar_trans = lidar_trans[-(num_after * 2) :]

fig, ax = plt.subplots(2, 1, figsize=(6, 6), layout="constrained", sharex=True)
ax[0].set_aspect("equal", "box")
ax[1].set_aspect("equal", "box")

for i in range(2):
    ax[i].plot(imu_trans[:, 1], imu_trans[:, 2 * i], "bo-", label="IMU Poses")
    ax[i].plot(lidar_trans[:, 1], lidar_trans[:, 2 * i], "ro-", label="Lidar Poses")
    ax[i].set_xlabel("Y (m)")
    ax[i].set_ylabel("X (m)" if i == 0 else "Z (m)")
    ax[i].grid(True)

# ax[0].set_xlim([-32, -31])
# ax[0].set_ylim([33, 34])

plt.legend()
plt.savefig("scripts/view_integration.png")
plt.show()
