from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import sys
from typing import Any, Optional, cast, overload
from typing_extensions import Callable
from tqdm import tqdm

from evalio import datasets as ds, types as ty

sys.path.append(".")
from imu_sim import Array
from form import gtsam
import numpy as np

np.set_printoptions(precision=3, suppress=True)


@overload
def convert(v: ty.SO3) -> gtsam.Rot3: ...  # type: ignore


def convert(v: Any) -> Any:
    return gtsam.Rot3(x=v.qx, y=v.qy, z=v.qz, w=v.qw)


def timeit(
    func: Callable[..., Array | tuple[Array, gtsam.Unit3]],
) -> Callable[..., Array | tuple[Array, gtsam.Unit3]]:
    import time

    def wrapper(*args, **kwargs) -> Array | tuple[Array, gtsam.Unit3]:
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        if isinstance(result, tuple):
            if len(args) > 2:
                g_naive = args[2]
                cos = result[1].point3().dot(g_naive.point3())
                angle = np.arccos(cos) * 180.0 / np.pi
                result_str = f"{result[0]}, {angle:.3f}°"
            else:
                result_str = f"{result[0]}"
        else:
            result_str = f"{result}"

        print(
            f"[{func.__name__.split('_', 1)[1]:^20}]: {end - start:.4f}s, {result_str}"
        )
        return result

    return wrapper


# ------------------------- Summarize dataset ------------------------- #
StampedPose = tuple[ty.Stamp, ty.SE3]


@dataclass(kw_only=True)
class ImuWindow:
    mm: list[ty.ImuMeasurement]
    start: StampedPose
    end: StampedPose

    def preint(
        self,
        bias_accel: Array = np.zeros(3),
        bias_gyro: Array = np.zeros(3),
        gravity: Array = np.zeros(3),
    ) -> gtsam.PreintegratedCombinedMeasurements:
        params = gtsam.PreintegrationCombinedParams(gravity)

        bias = gtsam.ConstantBias(bias_accel, bias_gyro)
        pim = gtsam.PreintegratedCombinedMeasurements(params, bias)

        for i in range(1, len(self.mm)):
            dt = (self.mm[i].stamp - self.mm[i - 1].stamp).to_sec()
            if dt <= 1e-8:
                continue
            pim.integrateMeasurement(self.mm[i].accel, self.mm[i].gyro, dt)

        return pim


def summarize_windows(
    dataset: ds.Dataset,
    every: float = 0.1,
    seconds: float = 20.0,
) -> tuple[list[ty.ImuMeasurement], list[ImuWindow]]:
    # Gather ground truth poses
    gt_og = dataset.ground_truth()

    first_imu = dataset.get_one_imu(0).stamp
    start_idx = 0
    while gt_og.stamps[start_idx] < first_imu:
        start_idx += 1

    # Keep a ground truth every `every` seconds
    gt = ty.Trajectory()
    prev_stamp = ty.Stamp.from_sec(0)
    for i in range(start_idx, len(gt_og)):
        # kludge to avoid floating point issues
        if (gt_og.stamps[i] - prev_stamp).to_sec() >= every - 1e-4:
            prev_stamp = gt_og.stamps[i]
            gt.append(gt_og.stamps[i], gt_og.poses[i])

        if (gt_og.stamps[i] - gt_og.stamps[start_idx]).to_sec() >= seconds:
            break

    # Gather IMU measurements into windows
    windows: list[ImuWindow] = []
    all_mm: list[ty.ImuMeasurement] = []

    curr_idx = 1
    curr = ImuWindow(mm=[], start=gt[0], end=gt[1])
    for mm in tqdm(dataset.imu()):
        all_mm.append(mm)
        # if it's too early, skip
        if mm.stamp < curr.start[0]:
            continue
        # if it's within the current window, add it
        elif mm.stamp < curr.end[0]:
            curr.mm.append(mm)
        # if it's past the current window, finalize and start a new one
        else:
            # Interpolate to get measurement at the exact end time
            mm_shifted = ty.ImuMeasurement(curr.end[0], accel=mm.accel, gyro=mm.gyro)
            curr.mm.append(mm_shifted)

            windows.append(curr)
            if curr_idx + 1 >= len(gt):
                break
            curr = ImuWindow(
                mm=[mm_shifted, mm], start=gt[curr_idx], end=gt[curr_idx + 1]
            )
            curr_idx += 1

    return all_mm, windows


# ------------------------- Gyro Estimation ------------------------- #
def estimate_gyro_graph(windows: list[ImuWindow]) -> Array:
    def gyro_error(
        pim: gtsam.PreintegratedCombinedMeasurements,
        R0: gtsam.Rot3,
        R1: gtsam.Rot3,
        this: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: gtsam.JacobianVector,
    ) -> Array:
        # Will have to dig into to see if I can fix
        gyro_bias = values.atVector(this.keys()[0])

        bias_delta = gyro_bias - pim.biasHat().gyroscope()
        corr = pim.preintegrated() + pim.preintegrated_H_biasOmega() @ bias_delta

        if jacobians is not None:
            # TODO: This jacobian isn't 100% correct, but it's close
            jacobians[0] = pim.preintegrated_H_biasOmega()[:3, :3]

        R1hat = R0.compose(gtsam.Rot3.Expmap(corr[:3]))
        return gtsam.Rot3.Logmap(R1.inverse().compose(R1hat))
        # I think this is the "gtsam" way to do it, but I'm going to leave the above for clarity
        # R1hat = R0.retract(preint[:3])
        # return R1.localCoordinates(R1hat)

    # Fill in graph
    graph = gtsam.NonlinearFactorGraph()
    for i, w in enumerate(windows[:-1]):
        c = gtsam.CustomFactor(
            gtsam.Isotropic.Sigma(3, 1e-2),
            [0],
            partial(
                gyro_error,
                w.preint(),
                convert(w.start[1].rot),
                convert(w.end[1].rot),
            ),
        )
        graph.push_back(c)

    # Fill in initial values
    values = gtsam.Values()
    values.insert_vector(0, np.zeros(3))

    # Optimize
    lm_params = gtsam.LevenbergMarquardtParams()
    # lm_params.setVerbosityLM("SUMMARY")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, lm_params)
    result = optimizer.optimize()
    return result.atVector(0)


def estimate_gyro_naive(sim: list[ty.ImuMeasurement]) -> Array:
    gyro_measurements = np.array([s.gyro for s in sim])
    return np.mean(gyro_measurements, axis=0)


def estimate_gyro_linear(sim: list[ImuWindow]) -> Array:
    preints = [w.preint() for w in windows[:-1]]

    # Collect H matrix
    H = np.vstack([pim.preintegrated_H_biasOmega()[:3, :3] for pim in preints])

    # Collect b vector
    r = np.concatenate(
        [
            gtsam.Rot3.Logmap(
                convert(windows[i].end[1].rot).inverse()
                * convert(windows[i].start[1].rot)
            )
            for i in range(len(windows) - 1)
        ]
    )
    xi = np.concatenate([pim.preintegrated()[:3] for pim in preints])
    b = r + xi

    # Solve for bias
    bias = np.linalg.lstsq(H, -b, rcond=None)[0]
    bias = cast(Array, bias)
    return bias


# ------------------------- Accel / Gravity Estimation ------------------------- #
def graph_estimation(
    error: Callable[
        [
            gtsam.PreintegratedCombinedMeasurements,
            StampedPose,
            StampedPose,
            StampedPose,
            gtsam.CustomFactor,
            gtsam.Values,
            gtsam.JacobianVector,
        ],
        Array,
    ],
    windows: list[ImuWindow],
    gyro_bias: Array,
    gravity_init: gtsam.Unit3,
    bias_prior: Optional[float] = None,
) -> tuple[Array, gtsam.Unit3]:
    # Fill in graph
    graph = gtsam.NonlinearFactorGraph()
    for i in range(1, len(windows) - 1):
        prev = windows[i - 1].start
        curr = windows[i].start
        next = windows[i + 1].start
        pim = windows[i].preint(bias_gyro=gyro_bias)

        c = gtsam.CustomFactor(
            gtsam.Isotropic.Sigma(3, 1e-2),
            [0, 1],
            partial(error, pim, prev, curr, next),
        )
        graph.push_back(c)

    # add prior on accel bias
    if bias_prior is not None:
        graph.addPriorVector(
            0,
            np.zeros(3),
            gtsam.Isotropic.Sigma(3, bias_prior),
        )

    # Fill in initial values
    values = gtsam.Values()
    # Accel bias
    values.insert_vector(0, np.zeros(3))
    # Gravity
    values.insert_unit3(1, gravity_init)

    # Optimize
    lm_params = gtsam.LevenbergMarquardtParams()
    # lm_params.setVerbosityLM("SUMMARY")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, lm_params)
    result = optimizer.optimize()
    return result.atVector(0), result.atUnit3(1)


def estimate_accel_p_only(
    windows: list[ImuWindow],
    gyro_bias: Array,
    grav_init: gtsam.Unit3,
    bias_prior: Optional[float] = None,
) -> tuple[Array, gtsam.Unit3]:
    def error(
        pim: gtsam.PreintegratedCombinedMeasurements,
        prev: StampedPose,
        curr: StampedPose,
        next: StampedPose,
        this: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: gtsam.JacobianVector,
    ) -> Array:
        # Will have to dig into to see if I can fix
        accel_bias = values.atVector(this.keys()[0])
        gravity = values.atUnit3(this.keys()[1])

        R0 = convert(curr[1].rot)
        p0 = curr[1].trans
        p1 = next[1].trans
        v0 = (next[1].trans - prev[1].trans) / (next[0] - prev[0]).to_sec()

        corr = pim.biasCorrectedDelta(gtsam.ConstantBias(accel_bias, gyro_bias))

        dt = pim.deltaTij()
        grav_scale = 0.5 * dt**2 * 9.81

        if jacobians is not None:
            # Position in the middle
            jacobians[0] = R0.matrix() @ pim.preintegrated_H_biasAcc()[3:6]
            jacobians[1] = np.eye(3) * grav_scale @ gravity.basis()

        p1hat = R0.rotate(corr[3:6]) + p0 + v0 * dt + grav_scale * gravity.point3()
        return p1hat - p1

    return graph_estimation(error, windows, gyro_bias, grav_init, bias_prior)


def estimate_accel_v_only(
    windows: list[ImuWindow],
    gyro_bias: Array,
    grav_init: gtsam.Unit3,
    bias_prior: Optional[float] = None,
) -> tuple[Array, gtsam.Unit3]:
    def error(
        pim: gtsam.PreintegratedCombinedMeasurements,
        prev: StampedPose,
        curr: StampedPose,
        next: StampedPose,
        this: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: gtsam.JacobianVector,
    ) -> Array:
        # Will have to dig into to see if I can fix
        accel_bias = values.atVector(this.keys()[0])
        gravity = values.atUnit3(this.keys()[1])

        R0 = convert(curr[1].rot)
        v0 = (curr[1].trans - prev[1].trans) / (curr[0] - prev[0]).to_sec()
        v1 = (next[1].trans - curr[1].trans) / (next[0] - curr[0]).to_sec()
        if (next[0] - curr[0]).to_sec() == 0:
            print(next[0], curr[0], prev[0])
            print((next[0] - curr[0]).to_sec(), next[0], curr[0], next[0] - curr[0])
            raise ValueError("Zero time difference in velocity computation")

        corr = pim.biasCorrectedDelta(gtsam.ConstantBias(accel_bias, gyro_bias))

        dt = pim.deltaTij()
        grav_scale = dt * 9.81

        if jacobians is not None:
            # Position in the middle
            jacobians[0] = R0.matrix() @ pim.preintegrated_H_biasAcc()[6:9]
            jacobians[1] = np.eye(3) * grav_scale @ gravity.basis()

        v1hat = R0.rotate(corr[6:9]) + v0 + gravity.point3() * grav_scale
        return v1hat - v1

    return graph_estimation(error, windows, gyro_bias, grav_init, bias_prior)


def estimate_accel_no_vp(
    windows: list[ImuWindow],
    gyro_bias: Array,
    grav_init: gtsam.Unit3,
    bias_prior: Optional[float] = None,
) -> tuple[Array, gtsam.Unit3]:
    def error(
        pim: gtsam.PreintegratedCombinedMeasurements,
        prev: StampedPose,
        start: StampedPose,
        end: StampedPose,
        this: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: gtsam.JacobianVector,
    ) -> Array:
        # Will have to dig into to see if I can fix
        accel_bias = values.atVector(this.keys()[0])
        gravity = values.atUnit3(this.keys()[1])

        R0 = convert(start[1].rot)

        corr = pim.biasCorrectedDelta(gtsam.ConstantBias(accel_bias, gyro_bias))

        dt = pim.deltaTij()
        grav_scale = 0.5 * dt**2 * 9.81

        if jacobians is not None:
            # Position in the middle
            jacobians[0] = R0.matrix() @ pim.preintegrated_H_biasAcc()[3:6]
            jacobians[1] = np.eye(3) * grav_scale @ gravity.basis()

        return R0.rotate(corr[3:6]) + grav_scale * gravity.point3()

    return graph_estimation(error, windows, gyro_bias, grav_init, bias_prior)


def estimate_accel_naive(
    mm: list[ty.ImuMeasurement], init: ty.SE3
) -> tuple[Array, gtsam.Unit3]:
    accel_measurements = np.array([s.accel for s in mm])
    gravity_unnorm = -np.mean(accel_measurements, axis=0)
    gravity = gravity_unnorm / np.linalg.norm(gravity_unnorm)

    accel = 9.81 * gravity - gravity_unnorm

    gravity = convert(init.rot).rotate(gravity)

    return accel, gtsam.Unit3(gravity)


if __name__ == "__main__":
    # data = ds.OxfordSpires.blenheim_palace_05
    # length = 10.0
    # every = 0.1

    data = ds.NewerCollege2020.short_experiment
    length = 80.0
    every = 0.2

    one_sec_imu = int(data.imu_params().rate)

    # TODO: Try plotting over a range of seconds to see how things do
    # Can just index windows iteratively to see how it does
    all_mm, windows = summarize_windows(
        data,
        every=every,
        seconds=length,
    )

    print()
    print(f"Dataset: {data.full_name}")
    delta = windows[0].start[1].inverse() * windows[-1].end[1]
    print(f"Amount Moved: {delta.trans}")
    print(f"Number of Windows: {len(windows)}")
    print()

    timeit(estimate_gyro_naive)(all_mm[:one_sec_imu])
    timeit(estimate_gyro_graph)(windows)
    gyro_bias = timeit(estimate_gyro_linear)(windows)
    gyro_bias = cast(Array, gyro_bias)
    print()

    b_naive, g_naive = timeit(estimate_accel_naive)(
        all_mm[:one_sec_imu], windows[0].start[1]
    )
    timeit(estimate_accel_p_only)(windows, gyro_bias, g_naive, 1e-1)
    timeit(estimate_accel_v_only)(windows, gyro_bias, g_naive, 1e-1)
    timeit(estimate_accel_p_only)(windows, gyro_bias, g_naive)
    timeit(estimate_accel_v_only)(windows, gyro_bias, g_naive)

    # ------------------------- Plot ------------------------- #
    one_sec_windows = int(1.0 / every) * 30
    priors = [None, 1e0, 0.5, 1e-1]

    bias = np.zeros((len(priors), len(windows) - one_sec_windows, 3))
    grav = np.zeros((len(priors), len(windows) - one_sec_windows, 3))

    for j, p in tqdm(enumerate(priors), total=len(priors)):
        for i in range(one_sec_windows, len(windows)):
            b, g = estimate_accel_p_only(windows[:i], gyro_bias, g_naive, p)
            bias[j, i - one_sec_windows] = b
            grav[j, i - one_sec_windows] = g.point3()

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(2, 3, figsize=(12, 6), sharex=True, layout="constrained")

    t = np.arange(bias.shape[1]) * every

    for i in range(3):
        ax[0, i].axhline(
            y=b_naive[i], color="r", linestyle="--", label="Naive Estimate"
        )
        ax[0, i].set_title(f"Accel Bias {['X', 'Y', 'Z'][i]}")
        ax[0, i].grid()

        ax[1, i].axhline(
            y=g_naive.point3()[i], color="r", linestyle="--", label="Naive Estimate"
        )
        ax[1, i].set_title(f"Gravity {['X', 'Y', 'Z'][i]}")
        ax[1, i].grid()

        for j in range(len(priors)):
            ax[0, i].plot(t, bias[j, :, i], label="Prior: " + str(priors[j]))
            ax[1, i].plot(t, grav[j, :, i])

    ax[0, 0].legend()

    plt.savefig(f"scripts/{data.full_name.replace('/', '_')}_bg_convergence.png")
