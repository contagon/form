from functools import partial
import sys
from typing import cast
from typing_extensions import Callable

sys.path.append(".")
from imu_sim import ImuSimulation, Array, Window
from form import gtsam
import numpy as np


def timeit(func: Callable[[ImuSimulation], Array]) -> Callable[[ImuSimulation], Array]:
    import time

    def wrapper(sim: ImuSimulation) -> Array:
        start = time.time()
        result = func(sim)
        end = time.time()
        print(f"[{func.__name__.split('_', 1)[1]:^20}]: {end - start:.4f}s, {result}")
        return result

    return wrapper


def estimate_full_graph(sim: ImuSimulation) -> Array:
    def gyro_error_full(
        window: Window,
        R0: gtsam.Rot3,
        R1: gtsam.Rot3,
        this: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: gtsam.JacobianVector,
    ) -> Array:
        # Will have to dig into to see if I can fix
        gyro_bias = values.atVector(this.keys()[0])
        pim = window.preint(bias_gyro=gyro_bias)

        # It doesn't appear the linearization really changes at all!
        # if window.states[0].stamp == 0.0:
        # print(pim.preintegrated_H_biasOmega()[:3, :3])

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

    windows = sim.summarize(every=0.1)

    # Fill in graph
    graph = gtsam.NonlinearFactorGraph()
    for i, w in enumerate(windows[:-1]):
        c = gtsam.CustomFactor(
            gtsam.Isotropic.Sigma(3, 1e-2),
            [0],
            partial(
                gyro_error_full,
                w,
                w.states[0].rotation,
                windows[i + 1].states[0].rotation,
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


def estimate_first_order_graph(sim: ImuSimulation) -> Array:
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
            # TODO: Need to evaluate how much of an effect this linearization has
            # Would it be fast enough to perform the integration at each iteration?
            jacobians[0] = pim.preintegrated_H_biasOmega()[:3, :3]

            # TODO: This jacobian isn't 100% correct, but it's close
            # corr_D_bias = pim.preintegrated_H_biasOmega()[:3, :3]
            # exp_D_corr = gtsam.Rot3.ExpmapDerivative(corr[:3])
            # R_D_exp = R_delta.matrix()
            # log_D_R = gtsam.Rot3.LogmapDerivative(residual)
            # jac = log_D_R @ R_D_exp @ exp_D_corr @ corr_D_bias

        R1hat = R0.compose(gtsam.Rot3.Expmap(corr[:3]))
        return gtsam.Rot3.Logmap(R1.inverse().compose(R1hat))
        # I think this is the "gtsam" way to do it, but I'm going to leave the above for clarity
        # R1hat = R0.retract(preint[:3])
        # return R1.localCoordinates(R1hat)

    windows = sim.summarize(every=0.1)

    # Fill in graph
    graph = gtsam.NonlinearFactorGraph()
    for i, w in enumerate(windows[:-1]):
        c = gtsam.CustomFactor(
            gtsam.Isotropic.Sigma(3, 1e-2),
            [0],
            partial(
                gyro_error,
                w.preint(),
                w.states[0].rotation,
                windows[i + 1].states[0].rotation,
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


def estimate_naive(sim: ImuSimulation) -> Array:
    gyro_measurements = np.array([s.gyro for s in sim.state])
    return np.mean(gyro_measurements, axis=0)


def estimate_linear(sim: ImuSimulation) -> Array:
    windows = sim.summarize(every=0.1)
    preints = [w.preint() for w in windows[:-1]]

    # Collect H matrix
    H = np.vstack([pim.preintegrated_H_biasOmega()[:3, :3] for pim in preints])

    # Collect b vector
    r = np.concatenate(
        [
            gtsam.Rot3.Logmap(
                windows[i + 1].states[0].rotation.inverse()
                * windows[i].states[0].rotation
            )
            for i in range(len(windows) - 1)
        ]
    )
    xi = np.concatenate([pim.preintegrated()[:3] for pim in preints])
    bhat = preints[0].biasHat().gyroscope()
    b = r + xi - H @ bhat

    # Solve for bias
    bias = np.linalg.lstsq(H, -b, rcond=None)[0]
    bias = cast(Array, bias)
    return bias


sim = ImuSimulation(
    imu_rate=100.0,
    # accel_gen=lambda t: np.array([0.2, 0.0, 0.0]),
    # accel_noise_sigma=0.0,
    gyro_gen=lambda t: np.array([-0.4, 0.3, 0.2]) * np.sin(t / 1.3),
    gyro_noise_sigma=1e-4,
    gyro_bias=np.array([0.01, -0.02, 0.5]),
    total_time=10.0,
)

print("Ground truth gyro bias:", sim.gyro_bias)
timeit(estimate_full_graph)(sim)
timeit(estimate_first_order_graph)(sim)
timeit(estimate_linear)(sim)
timeit(estimate_naive)(sim)

print("---- finished ----")
