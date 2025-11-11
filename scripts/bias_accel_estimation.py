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

    def __call__(
        self, sim: ImuSimulation, *args, **kwargs
    ) -> tuple[Array, gtsam.Unit3]: ...


def timeit(func: EstimationMethod) -> EstimationMethod:
    import time

    def wrapper(sim: ImuSimulation, *args, **kwargs) -> tuple[Array, gtsam.Unit3]:
        start = time.time()
        accel, grav = func(sim, *args, **kwargs)
        end = time.time()

        cos_grav_angle = (
            grav.point3()
            @ sim.gravity
            / (np.linalg.norm(grav.point3()) * np.linalg.norm(gravity))
        )
        grav_angle = np.arccos(cos_grav_angle) * 180 / np.pi

        bias_error = accel - sim.accel_bias
        bias_error = np.linalg.norm(bias_error)

        name = func.__name__.split("_", 1)[1]
        try:
            args = " ".join([f"{a:.1f}" for a in args])
        except:
            args = ""

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
) -> tuple[Array, gtsam.Unit3]:
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

    # add prior on accel bias
    if bias_prior is not None:
        graph.addPriorVector(
            0,
            np.zeros(3),
            gtsam.Isotropic.Sigma(3, 1e-2),
        )

    accel = np.array([s.accel for s in sim.state])
    gravity_init = -np.mean(accel, axis=0)
    gravity_init = gtsam.Unit3(gravity_init)

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


# ------------------------- Methods that use GT Velocity ------------------------- #
def estimate_graph_gt_vel_p_only(
    sim: ImuSimulation,
    bias_prior: Optional[float] = None,
) -> tuple[Array, gtsam.Unit3]:
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
        gravity = values.atUnit3(this.keys()[1])

        R0 = start.rotation
        v0 = start.velocity
        p0 = start.position
        p1 = end.position

        # TODO: Using gt bias for gyro here. Eventually will want to switch to estimated
        corr = pim.biasCorrectedDelta(gtsam.ConstantBias(accel_bias, sim.gyro_bias))

        dt = pim.deltaTij()
        grav_scale = 0.5 * dt**2 * 9.81

        if jacobians is not None:
            # Position in the middle
            jacobians[0] = R0.matrix() @ pim.preintegrated_H_biasAcc()[3:6]
            jacobians[1] = np.eye(3) * grav_scale @ gravity.basis()

        # return R0.rotate(corr[3:6]) + 0.5 * gravity * dt**2

        p1hat = R0.rotate(corr[3:6]) + p0 + v0 * dt + grav_scale * gravity.point3()
        return p1hat - p1

    return graph_estimation(error, sim, bias_prior)


def estimate_graph_gt_vel_v_only(
    sim: ImuSimulation,
    bias_prior: Optional[float] = None,
) -> tuple[Array, gtsam.Unit3]:
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
        gravity = values.atUnit3(this.keys()[1])

        R0 = start.rotation
        v0 = start.velocity
        v1 = end.velocity

        # TODO: Using gt bias for gyro here. Eventually will want to switch to estimated
        corr = pim.biasCorrectedDelta(gtsam.ConstantBias(accel_bias, sim.gyro_bias))

        dt = pim.deltaTij()
        grav_scale = dt * 9.81

        if jacobians is not None:
            # Position in the middle
            jacobians[0] = R0.matrix() @ pim.preintegrated_H_biasAcc()[6:9]
            jacobians[1] = np.eye(3) * dt * 9.81 @ gravity.basis()

        v1hat = R0.rotate(corr[6:9]) + v0 + gravity.point3() * grav_scale
        return v1hat - v1

    return graph_estimation(error, sim, bias_prior)


# ------------------------- Methods that estimate velocity ------------------------- #
def estimate_graph_est_vel_p_only(
    sim: ImuSimulation,
    bias_prior: Optional[float] = None,
) -> tuple[Array, gtsam.Unit3]:
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
        gravity = values.atUnit3(this.keys()[1])

        R0 = curr.rotation
        p0 = curr.position
        p1 = next.position
        v0 = (next.position - prev.position) / (2 * pim.deltaTij())

        # TODO: Using gt bias for gyro here. Eventually will want to switch to estimated
        corr = pim.biasCorrectedDelta(gtsam.ConstantBias(accel_bias, sim.gyro_bias))

        dt = pim.deltaTij()
        grav_scale = 0.5 * dt**2 * 9.81

        if jacobians is not None:
            # Position in the middle
            jacobians[0] = R0.matrix() @ pim.preintegrated_H_biasAcc()[3:6]
            jacobians[1] = np.eye(3) * grav_scale @ gravity.basis()

        p1hat = R0.rotate(corr[3:6]) + p0 + v0 * dt + grav_scale * gravity.point3()
        return p1hat - p1

    return graph_estimation(error, sim, bias_prior)


def estimate_graph_est_vel_v_only(
    sim: ImuSimulation,
    bias_prior: Optional[float] = None,
) -> tuple[Array, gtsam.Unit3]:
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
        gravity = values.atUnit3(this.keys()[1])

        R0 = curr.rotation
        v0 = (curr.position - prev.position) / pim.deltaTij()
        v1 = (next.position - curr.position) / pim.deltaTij()

        # TODO: Using gt bias for gyro here. Eventually will want to switch to estimated
        corr = pim.biasCorrectedDelta(gtsam.ConstantBias(accel_bias, sim.gyro_bias))

        dt = pim.deltaTij()
        grav_scale = dt * 9.81

        if jacobians is not None:
            # Position in the middle
            jacobians[0] = R0.matrix() @ pim.preintegrated_H_biasAcc()[6:9]
            jacobians[1] = np.eye(3) * grav_scale @ gravity.basis()

        v1hat = R0.rotate(corr[6:9]) + v0 + gravity.point3() * grav_scale
        return v1hat - v1

    return graph_estimation(error, sim, bias_prior)


def estimate_graph_no_vp(
    sim: ImuSimulation,
    bias_prior: Optional[float] = None,
) -> tuple[Array, gtsam.Unit3]:
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
        gravity = values.atUnit3(this.keys()[1])

        R0 = start.rotation

        corr = pim.biasCorrectedDelta(gtsam.ConstantBias(accel_bias, sim.gyro_bias))

        dt = pim.deltaTij()
        grav_scale = 0.5 * dt**2 * 9.81

        if jacobians is not None:
            # Position in the middle
            jacobians[0] = R0.matrix() @ pim.preintegrated_H_biasAcc()[3:6]
            jacobians[1] = np.eye(3) * grav_scale @ gravity.basis()

        return R0.rotate(corr[3:6]) + grav_scale * gravity.point3()

    return graph_estimation(error, sim, bias_prior)


def estimate_analytic(sim: ImuSimulation) -> tuple[Array, gtsam.Unit3]:
    windows = sim.summarize(every=0.1)
    preints = [w.preint(bias_gyro=sim.gyro_bias) for w in windows[1:-1]]

    # TODO: I'm not handling bias_hat properly here, it should be a negative correction later
    # Shouldn't matter as it's zero for now

    R0s = [w.states[0].rotation.matrix() for w in windows[1:-1]]

    # First form our linear system
    H = np.concatenate(
        [R @ p.preintegrated_H_biasAcc()[3:6] for R, p in zip(R0s, preints)], axis=0
    )
    G = np.concatenate([0.5 * p.deltaTij() ** 2 * np.eye(3) for p in preints], axis=0)
    Alst = np.concatenate([H, G], axis=1)

    Asq = Alst.T @ Alst
    print(np.linalg.matrix_rank(Alst))
    print(np.linalg.cond(Alst))
    print(np.linalg.cond(Asq))
    print(Asq)
    # print(Alst[:24])

    init_bias = gtsam.ConstantBias(np.zeros(3), sim.gyro_bias)
    xi = np.concatenate(
        [R @ p.biasCorrectedDelta(init_bias)[3:6] for R, p in zip(R0s, preints)],
        axis=0,
    )
    p0 = np.concatenate([w.states[0].position for w in windows[1:-1]])
    p1 = np.concatenate([w.states[1].position for w in windows[1:-1]])
    dt_v0 = np.concatenate(
        [
            p.deltaTij()
            * (next.states[0].velocity - prev.states[0].velocity)
            / (2 * p.deltaTij())
            for p, prev, next in zip(preints, windows[:-2], windows[2:])
        ]
    )
    blst = xi + p0 + dt_v0 - p1
    # print(blst[:24])

    #     # Now solve polynomial for constrained opt
    #     Abig = Alst.T @ Alst
    #     A = 2 * Abig[:3, :3]
    #     B = 2 * Abig[:3, 3:6]
    #     D = 2 * Abig[3:6, 3:6]
    #     m = -2 * blst.T @ A

    #     S = D - B.T @ np.linalg.inv(A) @ B
    #     U = np.linalg.trace(S) * np.eye(3) - S
    #     Apow = lambda S: np.linalg.det(S) * np.linalg.inv(S)
    #     X = 2 * Apow(S) + U @ U
    #     Y = Apow(S) @ U + U @ Apow(S)

    #     coeffs = np.zeros(6)

    #     Ainv = np.linalg.inv(A)
    #     mat = Ainv @ B @ B.T @ Ainv.T
    #     coeffs[4] = m.T

    #     print(xi.shape)


# def estimate_naive(sim: ImuSimulation, *args) -> tuple[Array, gtsam.Unit3]:
#     accel_measurements = np.array([s.accel for s in sim.state])
#     gravity_unnorm = -np.mean(accel_measurements, axis=0)
#     gravity = gravity_unnorm / np.linalg.norm(gravity_unnorm)

#     return np.zeros(3), gtsam.Unit3(gravity)


def estimate_naive(sim: ImuSimulation, init: gtsam.Rot3) -> tuple[Array, gtsam.Unit3]:
    accel_measurements = np.array([s.accel for s in sim.state])
    gravity_unnorm = -np.mean(accel_measurements, axis=0)
    gravity = gravity_unnorm / np.linalg.norm(gravity_unnorm)

    accel = 9.81 * gravity - gravity_unnorm

    gravity = init.unrotate(gravity)

    return accel, gtsam.Unit3(gravity)


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
    # gyro_noise_sigma=1e-4,
    # gyro_bias=np.array([0.01, -0.02, 0.5]),
    total_time=20.0,
)

print("Ground truths:", sim.gravity, sim.accel_bias)
timeit(estimate_naive)(sim, sim.state[0].rotation)
timeit(estimate_graph_gt_vel_p_only)(sim)
timeit(estimate_graph_est_vel_p_only)(sim)
timeit(estimate_graph_gt_vel_v_only)(sim)
timeit(estimate_graph_est_vel_v_only)(sim)
# timeit(estimate_graph_no_vp)(sim)

# estimate_analytic(sim)

# print(estimate_analytic(sim))

print("---- finished ----")
