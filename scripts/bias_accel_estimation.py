from functools import partial
import sys
from typing import Callable, Optional, Protocol

sys.path.append(".")
from bias_jacobian_testing import nderiv  # noqa: F401
from imu_sim import ImuSimulation, Array, State
from form import gtsam
import numpy as np


class EstimationMethod(Protocol):
    @property
    def __name__(self) -> str: ...

    def __call__(self, sim: ImuSimulation, *args, **kwargs) -> tuple[Array, Array]: ...


def timeit(func: EstimationMethod) -> EstimationMethod:
    import time

    def wrapper(sim: ImuSimulation, *args, **kwargs) -> tuple[Array, Array]:
        start = time.time()
        accel, grav = func(sim, *args, **kwargs)
        end = time.time()

        cos_grav_angle = (
            grav @ sim.gravity / (np.linalg.norm(grav) * np.linalg.norm(gravity))
        )
        grav_angle = np.arccos(cos_grav_angle) * 180 / np.pi

        bias_error = accel - sim.accel_bias
        bias_error = np.linalg.norm(bias_error)

        name = func.__name__.split("_", 1)[1]
        args = " ".join([f"{a:.1f}" for a in args])

        print(
            f"[{name:^20}][{args:^6}]: {end - start:.3f}s, g: {grav_angle:.3f}°, b: {bias_error:.3f}"
        )
        return accel, grav

    return wrapper


def graph_estimation(
    error: Callable[
        [
            gtsam.PreintegratedCombinedMeasurements,
            State,
            State,
            State,
            gtsam.CustomFactor,
            gtsam.Values,
            gtsam.JacobianVector,
        ],
        Array,
    ],
    sim: ImuSimulation,
    bias_prior: Optional[float] = None,
) -> tuple[Array, Array]:
    def gravity_magnitude_prior(
        this: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: gtsam.JacobianVector,
    ) -> Array:
        gravity = values.atVector(this.keys()[0])
        norm2 = gravity @ gravity
        if jacobians is not None:
            if norm2 > 1e-8:
                jacobians[0] = 2 * gravity.reshape(1, 3)
            else:
                jacobians[0] = np.zeros((1, 3))
        return np.array([norm2 - 9.81**2])

    windows = sim.summarize(every=0.1)

    # Fill in graph
    graph = gtsam.NonlinearFactorGraph()
    for i in range(1, len(windows) - 1):
        prev = windows[i - 1].states[0]
        curr = windows[i].states[0]
        next = windows[i + 1].states[0]
        pim = windows[i].preint(bias_gyro=sim.gyro_bias)
        c = gtsam.CustomFactor(
            gtsam.Isotropic.Sigma(3, 1e-2),
            [0, 1],
            partial(error, pim, prev, curr, next),
        )
        graph.push_back(c)

    # Add constraint on gravity magnitude
    c = gtsam.CustomFactor(
        gtsam.Constrained.All(1),
        [1],
        gravity_magnitude_prior,
    )
    graph.push_back(c)

    # add prior on accel bias
    if bias_prior is not None:
        graph.addPriorVector(
            0,
            np.zeros(3),
            gtsam.Isotropic.Sigma(3, 1e-2),
        )

    accel = np.array([s.accel for s in sim.state])
    gravity_init = -np.mean(accel, axis=0)
    gravity_init *= 9.81 / np.linalg.norm(gravity_init)

    # Fill in initial values
    values = gtsam.Values()
    # Accel bias
    values.insert_vector(0, np.zeros(3))
    # Gravity
    values.insert_vector(1, gravity_init)

    # Optimize
    lm_params = gtsam.LevenbergMarquardtParams()
    # lm_params.setVerbosityLM("SUMMARY")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, lm_params)
    result = optimizer.optimize()

    return result.atVector(0), result.atVector(1)


# ------------------------- Methods that use GT Velocity ------------------------- #
def estimate_graph_gt_vel_p_only(
    sim: ImuSimulation,
    bias_prior: Optional[float] = None,
) -> tuple[Array, Array]:
    def error(
        pim: gtsam.PreintegratedCombinedMeasurements,
        prev: State,
        start: State,
        end: State,
        this: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: gtsam.JacobianVector,
    ) -> Array:
        # Will have to dig into to see if I can fix
        accel_bias = values.atVector(this.keys()[0])
        gravity = values.atVector(this.keys()[1])

        R0 = start.rotation
        v0 = start.velocity
        p0 = start.position
        p1 = end.position

        # TODO: Using gt bias for gyro here. Eventually will want to switch to estimated
        corr = pim.biasCorrectedDelta(gtsam.ConstantBias(accel_bias, sim.gyro_bias))

        dt = pim.deltaTij()

        if jacobians is not None:
            # Position in the middle
            jacobians[0] = R0.matrix() @ pim.preintegrated_H_biasAcc()[3:6]
            jacobians[1] = np.eye(3) * (0.5 * dt**2)

        # return R0.rotate(corr[3:6]) + 0.5 * gravity * dt**2

        p1hat = R0.rotate(corr[3:6]) + p0 + v0 * dt + 0.5 * gravity * dt**2
        return p1hat - p1

    return graph_estimation(error, sim, bias_prior)


def estimate_graph_gt_vel_v_only(
    sim: ImuSimulation,
    bias_prior: Optional[float] = None,
) -> tuple[Array, Array]:
    def error(
        pim: gtsam.PreintegratedCombinedMeasurements,
        prev: State,
        start: State,
        end: State,
        this: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: gtsam.JacobianVector,
    ) -> Array:
        # Will have to dig into to see if I can fix
        accel_bias = values.atVector(this.keys()[0])
        gravity = values.atVector(this.keys()[1])

        R0 = start.rotation
        v0 = start.velocity
        v1 = end.velocity

        # TODO: Using gt bias for gyro here. Eventually will want to switch to estimated
        corr = pim.biasCorrectedDelta(gtsam.ConstantBias(accel_bias, sim.gyro_bias))

        dt = pim.deltaTij()

        if jacobians is not None:
            # Position in the middle
            jacobians[0] = R0.matrix() @ pim.preintegrated_H_biasAcc()[6:9]
            jacobians[1] = np.eye(3) * dt

        v1hat = R0.rotate(corr[6:9]) + v0 + gravity * dt
        return v1hat - v1

    return graph_estimation(error, sim, bias_prior)


# ------------------------- Methods that estimate velocity ------------------------- #
def estimate_graph_est_vel_p_only(
    sim: ImuSimulation,
    bias_prior: Optional[float] = None,
) -> tuple[Array, Array]:
    def error(
        pim: gtsam.PreintegratedCombinedMeasurements,
        prev: State,
        curr: State,
        next: State,
        this: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: gtsam.JacobianVector,
    ) -> Array:
        # Will have to dig into to see if I can fix
        accel_bias = values.atVector(this.keys()[0])
        gravity = values.atVector(this.keys()[1])

        R0 = curr.rotation
        p0 = curr.position
        p1 = next.position
        v0 = (next.position - prev.position) / (2 * pim.deltaTij())

        # TODO: Using gt bias for gyro here. Eventually will want to switch to estimated
        corr = pim.biasCorrectedDelta(gtsam.ConstantBias(accel_bias, sim.gyro_bias))

        dt = pim.deltaTij()

        if jacobians is not None:
            # Position in the middle
            jacobians[0] = R0.matrix() @ pim.preintegrated_H_biasAcc()[3:6]
            jacobians[1] = np.eye(3) * (0.5 * dt**2)

        # return R0.rotate(corr[3:6]) + 0.5 * gravity * dt**2

        p1hat = R0.rotate(corr[3:6]) + p0 + v0 * dt + 0.5 * gravity * dt**2
        return p1hat - p1

    return graph_estimation(error, sim, bias_prior)


def estimate_graph_est_vel_v_only(
    sim: ImuSimulation,
    bias_prior: Optional[float] = None,
) -> tuple[Array, Array]:
    def error(
        pim: gtsam.PreintegratedCombinedMeasurements,
        prev: State,
        curr: State,
        next: State,
        this: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: gtsam.JacobianVector,
    ) -> Array:
        # Will have to dig into to see if I can fix
        accel_bias = values.atVector(this.keys()[0])
        gravity = values.atVector(this.keys()[1])

        R0 = curr.rotation
        v0 = (curr.position - prev.position) / pim.deltaTij()
        v1 = (next.position - curr.position) / pim.deltaTij()

        # TODO: Using gt bias for gyro here. Eventually will want to switch to estimated
        corr = pim.biasCorrectedDelta(gtsam.ConstantBias(accel_bias, sim.gyro_bias))

        dt = pim.deltaTij()
        if jacobians is not None:
            # Position in the middle
            jacobians[0] = R0.matrix() @ pim.preintegrated_H_biasAcc()[6:9]
            jacobians[1] = np.eye(3) * dt

        v1hat = R0.rotate(corr[6:9]) + v0 + gravity * dt
        return v1hat - v1

    return graph_estimation(error, sim, bias_prior)


def estimate_graph_no_vp(
    sim: ImuSimulation,
    bias_prior: Optional[float] = None,
) -> tuple[Array, Array]:
    def error(
        pim: gtsam.PreintegratedCombinedMeasurements,
        prev: State,
        start: State,
        end: State,
        this: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: gtsam.JacobianVector,
    ) -> Array:
        # Will have to dig into to see if I can fix
        accel_bias = values.atVector(this.keys()[0])
        gravity = values.atVector(this.keys()[1])

        R0 = start.rotation

        corr = pim.biasCorrectedDelta(gtsam.ConstantBias(accel_bias, sim.gyro_bias))

        dt = pim.deltaTij()

        if jacobians is not None:
            # Position in the middle
            jacobians[0] = R0.matrix() @ pim.preintegrated_H_biasAcc()[3:6]
            jacobians[1] = np.eye(3) * (0.5 * dt**2)

        return R0.rotate(corr[3:6]) + 0.5 * gravity * dt**2

    return graph_estimation(error, sim, bias_prior)


def estimate_naive(sim: ImuSimulation) -> tuple[Array, Array]:
    accel_measurements = np.array([s.accel for s in sim.state])
    gravity_unnorm = -np.mean(accel_measurements, axis=0)
    gravity = gravity_unnorm * 9.81 / np.linalg.norm(gravity_unnorm)

    return np.zeros(3), gravity


np.random.seed(0)
gravity = np.random.normal(size=3)
gravity *= 9.81 / np.linalg.norm(gravity)

sim = ImuSimulation(
    imu_rate=100.0,
    gravity=gravity,
    #### accelerometer
    accel_noise_sigma=1e-2,
    accel_gen=lambda t: np.array([0.2, 0.8, 0.1]) * np.cos(t / 2.0),
    accel_bias=np.array([0.4, -0.05, 0.2]),
    #### gyroscope
    gyro_gen=lambda t: np.array([-0.4, 0.3, 0.2]) * np.sin(t / 1.3),
    gyro_noise_sigma=1e-4,
    gyro_bias=np.array([0.01, -0.02, 0.5]),
    total_time=20.0,
)

print("Ground truths:", sim.gravity, sim.accel_bias)
timeit(estimate_naive)(sim)
timeit(estimate_graph_gt_vel_p_only)(sim)
timeit(estimate_graph_est_vel_p_only)(sim)
timeit(estimate_graph_gt_vel_v_only)(sim)
timeit(estimate_graph_est_vel_v_only)(sim)
timeit(estimate_graph_no_vp)(sim)

print("---- finished ----")
