# ruff: noqa: E731
from dataclasses import dataclass, field
import evalio  # noqa: F401
from form import gtsam
import numpy as np
from typing import Callable

from numpy.typing import NDArray

np.set_printoptions(precision=4, suppress=True)

Array = NDArray[np.float64]


@dataclass(kw_only=True)
class State:
    # ground truth values
    stamp: float
    rotation: gtsam.Rot3
    position: Array
    velocity: Array
    # measurements corrupted by noise and bias
    accel: Array
    gyro: Array


@dataclass(kw_only=True)
class Window:
    dt: float
    gravity: Array = field(default_factory=lambda: np.array([0, 0, -9.81]))
    states: list[State] = field(default_factory=list)

    def append(self, state: State):
        self.states.append(state)

    def preint(
        self, bias_accel: Array = np.zeros(3), bias_gyro: Array = np.zeros(3)
    ) -> gtsam.PreintegratedCombinedMeasurements:
        params = gtsam.PreintegrationCombinedParams(self.gravity)

        bias = gtsam.ConstantBias(bias_accel, bias_gyro)
        pim = gtsam.PreintegratedCombinedMeasurements(params, bias)

        for s in self.states:
            pim.integrateMeasurement(s.accel, s.gyro, self.dt)

        return pim


@dataclass(kw_only=True)
class ImuSimulation:
    gravity: Array = field(default_factory=lambda: np.array([0, 0, -9.81]))
    accel_bias: Array = field(default_factory=lambda: np.zeros(3))
    gyro_bias: Array = field(default_factory=lambda: np.zeros(3))
    accel_noise_sigma: float = 0.0
    gyro_noise_sigma: float = 0.0
    imu_rate: float = 100.0  # Hz
    accel_gen: Callable[[float], Array] | Array = field(
        default_factory=lambda: np.zeros(3)
    )  # in body frame
    gyro_gen: Callable[[float], Array] | Array = field(
        default_factory=lambda: np.zeros(3)
    )  # in body frame
    total_time: float = 2.0  # seconds

    # to be simulated
    state: list[State] = field(default_factory=list)

    @property
    def stamps(self) -> Array:
        return np.array([s.stamp for s in self.state])

    @property
    def t(self) -> Array:
        return np.array([s.position for s in self.state])

    @property
    def v(self) -> Array:
        return np.array([s.velocity for s in self.state])

    def summarize(self, every: float = 0.5) -> list[Window]:
        # make sure every is multiple of dt
        dt = 1.0 / self.imu_rate
        step = int(every / dt)

        windows: list[Window] = []

        # This organizes the windows to integrate from start of current measurement to start of next
        # The last one should consist of a single state at the end time
        for i, s in enumerate(self.state):
            if i % step == 0:
                windows.append(Window(dt=dt, gravity=self.gravity))

            windows[-1].append(s)

        return windows

    def __post_init__(self):
        self.state = [
            State(
                stamp=0.0,
                rotation=gtsam.Rot3.Identity(),
                position=np.zeros(3),
                velocity=np.zeros(3),
                accel=np.zeros(3),
                gyro=np.zeros(3),
            )
        ]

        if not callable(self.accel_gen):
            accel = self.accel_gen.copy()
            self.accel_gen = lambda t: accel

        if not callable(self.gyro_gen):
            gyro = self.gyro_gen.copy()
            self.gyro_gen = lambda t: gyro

        dt = 1.0 / self.imu_rate
        num_steps = int(self.total_time * self.imu_rate)

        for i in range(1, num_steps + 1):
            prev = self.state[-1]

            R_prev = prev.rotation
            v_prev = prev.velocity
            p_prev = prev.position

            true_accel = self.accel_gen(i * dt) - R_prev.rotate(self.gravity)
            true_gyro = self.gyro_gen(i * dt)

            noisy_accel = (
                true_accel
                + self.accel_bias
                + np.random.normal(scale=self.accel_noise_sigma, size=3)
            )
            noisy_gyro = (
                true_gyro
                + self.gyro_bias
                + np.random.normal(scale=self.gyro_noise_sigma, size=3)
            )

            R_new = R_prev.compose(gtsam.Rot3.Expmap(true_gyro * dt))
            a_world = R_prev.rotate(true_accel) + self.gravity
            v_new = v_prev + a_world * dt
            p_new = p_prev + v_prev * dt + 0.5 * a_world * dt * dt

            self.state.append(
                State(
                    stamp=i * dt,
                    rotation=R_new,
                    position=p_new,
                    velocity=v_new,
                    accel=noisy_accel,
                    gyro=noisy_gyro,
                )
            )


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    sim = ImuSimulation(
        accel_gen=lambda t: np.array([0.2, 0.0, 0.0]),
        # accel_noise_sigma=0.0,
        gyro_gen=lambda t: np.array([0.0, 0.0, 0.2]),
        # gyro_noise_sigma=0.0,
        total_time=10.0,
    )

    positions = sim.t
    velocities = sim.v

    fig, axs = plt.subplots(2, 3, sharex=True, figsize=(12, 6))
    labels = ["x", "y", "z"]

    for i in range(3):
        axs[0, i].plot(sim.stamps, velocities[:, i])
        axs[0, i].set_title(f"Velocity {labels[i]}")
        axs[0, i].set_ylabel("m/s")
        axs[0, i].grid(True)

        axs[1, i].plot(sim.stamps, positions[:, i])
        axs[1, i].set_title(f"Position {labels[i]}")
        axs[1, i].set_ylabel("m")
        axs[1, i].set_xlabel("time [s]")
        axs[1, i].grid(True)

    fig.suptitle("IMU Simulation")
    plt.tight_layout()
    plt.show()
