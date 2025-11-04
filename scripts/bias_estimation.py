# ruff: noqa: E731
import evalio  # noqa: F401
from form import gtsam
import numpy as np
from typing import Callable, TypeVar

from numpy.typing import NDArray

np.set_printoptions(precision=4, suppress=True)


def skew(xi: NDArray[np.float64]) -> NDArray[np.float64]:
    """Returns the skew symmetric matrix of a 3D vector."""
    return np.array(
        [
            [0, -xi[2], xi[1]],
            [xi[2], 0, -xi[0]],
            [-xi[1], xi[0], 0],
        ]
    )


T = TypeVar("T")


def nderiv(f: Callable[[T], NDArray[np.float64]], x: T, eps=1e-6):
    # Figure out the adder type
    if isinstance(x, np.ndarray):
        add_eps = lambda v, e: v + e
        size = len(x)
    elif isinstance(x, gtsam.Rot3):
        add_eps = lambda v, e: v.retract(e)
        size = 3
    elif isinstance(x, gtsam.Pose3):
        add_eps = lambda v, e: v.retract(e)
        size = 6
    else:
        raise NotImplementedError("Type not supported for nderiv")

    fx = f(x)
    jacobian = np.zeros((len(fx), size))
    for i in range(size):
        eps = np.zeros(size)
        eps[i] = 1e-6
        jacobian[:, i] = (f(add_eps(x, eps)) - fx) / 1e-6

    return fx, jacobian


np.random.seed(0)
measurements = [np.random.normal(size=6) for _ in range(25)]
mm_dt = 0.01

X1 = gtsam.Pose3(gtsam.Rot3.Identity(), np.random.normal(size=3))
X1 = gtsam.Pose3.Expmap(np.random.normal(size=6))


def preintegrate(
    *,
    gravity: NDArray[np.float64] = np.array([0, 0, -9.81]),
    x0: gtsam.Pose3 = gtsam.Pose3.Identity(),
    x1: gtsam.Pose3 = X1,
    v0: NDArray[np.float64] = np.zeros(3),
    v1: NDArray[np.float64] = np.zeros(3),
    b: NDArray[np.float64] = np.zeros(6),
) -> NDArray[np.float64]:
    params = gtsam.PreintegrationCombinedParams(gravity)
    pim = gtsam.PreintegratedCombinedMeasurements(params)

    for m in measurements:
        pim.integrateMeasurement(m[:3], m[3:], mm_dt)

    # This is equivalent to our manual implementation below
    # bcb = gtsam.ConstantBias(b[:3], b[3:])
    # factor = gtsam.CombinedImuFactor(0, 1, 2, 3, 4, 5, pim)
    # err = factor.evaluateError(x0, v0, x1, v1, bcb, bcb)[:9]
    # Unrotate last two errors - this rotation comes from left invariant pose error
    # Not proper to undo, but I believe it will cancel out in the end
    # err[3:6] = x1.rotation().rotate(err[3:6])
    # err[6:9] = x1.rotation().rotate(err[6:9])
    # print("gtsam error:\n", err)

    bias_delta = b - pim.biasHatVector()
    preint = (
        pim.preintegrated()
        + pim.preintegrated_H_biasAcc() @ bias_delta[:3]
        + pim.preintegrated_H_biasOmega() @ bias_delta[3:]
    )

    R0 = x0.rotation()
    p0 = x0.translation()
    R1 = x1.rotation()
    p1 = x1.translation()
    dt = pim.deltaTij()

    R1hat = R0.compose(gtsam.Rot3.Expmap(preint[:3]))
    p1hat = R0.rotate(preint[3:6]) + p0 + v0 * dt + 0.5 * params.n_gravity * dt * dt
    v1hat = R0.rotate(preint[6:9]) + v0 + params.n_gravity * dt

    err = np.zeros(9)
    err[:3] = gtsam.Rot3.Logmap(R1.inverse().compose(R1hat))
    err[3:6] = p1hat - p1
    err[6:9] = v1hat - v1

    print("preint error:\n", err)

    return err


preintegrate()

# print("Gravity Jacobian:")
# fx, jac = nderiv(lambda g: preintegrate(gravity=g), np.array([0, 0, -9.81]))
# print(jac)
# print()

# print("Bias Jacobian:")
# fx, jac = nderiv(lambda b: preintegrate(b=b), np.zeros(6))
# print(jac)
# print()

# print("start")
# gravity = np.array([0, 0, -9.81])
# params = gtsam.PreintegrationCombinedParams(gravity)
# print("params made")
# pim = gtsam.PreintegratedCombinedMeasurements(params)
# print("made")

# prev_t = None
# for imu in ds.NewerCollege2020.short_experiment.imu():
#     if prev_t is not None:
#         dt = (imu.stamp - prev_t).to_sec()
#         pim.integrateMeasurement(imu.accel, imu.gyro, dt)
#     prev_t = imu.stamp

#     print(pim.deltaTij())
