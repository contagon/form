#include "form/inertial/imu.hpp"
#include <cassert>
#include <deque>
#include <gtsam/base/Vector.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/NavState.h>
#include <optional>

namespace form {
// ------------------------- Misc helpers ------------------------- //
// TODO: Verify this function
[[nodiscard]] static gtsam::NavState
integrate(const gtsam::NavState &state, const Eigen::Vector3d &gyro,
          const Eigen::Vector3d &acc, const gtsam::imuBias::ConstantBias &bias,
          const Eigen::Vector3d &world_gravity, const double dt) noexcept {
  return state.update(
      acc - bias.accelerometer() + state.pose().rotation().unrotate(world_gravity),
      gyro - bias.gyroscope(), dt, boost::none, boost::none, boost::none);
}

[[nodiscard]] static double computeCosThetaHalf(const Eigen::Vector3d &a,
                                                const Eigen::Vector3d &b) noexcept {

  return std::sqrt((a.dot(b) + static_cast<double>(1)) / static_cast<double>(2));
}

[[nodiscard]] static Eigen::Vector3d
computeOmegaSinThetaHalf(const Eigen::Vector3d &a, const Eigen::Vector3d &b,
                         const double cosThetaHalf) noexcept {
  return a.cross(b) / (static_cast<double>(2) * cosThetaHalf);
}

[[nodiscard]] static gtsam::Rot3 naiveAlign(const Eigen::Vector3d &b,
                                            const Eigen::Vector3d &a,
                                            const double cos_theta_half) noexcept {
  const Eigen::Vector3d omega_sin_theta_half =
      computeOmegaSinThetaHalf(a, b, cos_theta_half);
  return gtsam::Rot3(cos_theta_half, omega_sin_theta_half.x(),
                     omega_sin_theta_half.y(), omega_sin_theta_half.z());
}

[[nodiscard]] static gtsam::Rot3 naiveAlign(const Eigen::Vector3d &b,
                                            const Eigen::Vector3d &a) noexcept {
  const double cos_theta_half = computeCosThetaHalf(a, b);
  return naiveAlign(b, a, cos_theta_half);
}

[[nodiscard]] static gtsam::Rot3 align(const Eigen::Vector3d &b,
                                       const Eigen::Vector3d &a) noexcept {
  const double cos_theta_half = computeCosThetaHalf(a, b);
  if (cos_theta_half >= 1e-3 - 1.0) {
    return naiveAlign(b, a, cos_theta_half);
  } else {
    int j = 0;
    a.array().abs().minCoeff(&j);
    Eigen::Vector3d e_j = Eigen::Vector3d::Zero();
    e_j(j) = 1.0;
    const gtsam::Rot3 c_R_a(0.0, e_j.x(), e_j.y(), e_j.z());
    return naiveAlign(b, c_R_a * a) * c_R_a;
  }
}

[[nodiscard]] gtsam::Pose3 interpolate(const std::deque<ImuState> &w_T_ks,
                                       const Stamp &stamp) noexcept {

  auto before = w_T_ks.cbegin();
  auto after = before + 1;
  for (; before != w_T_ks.cend() && after != w_T_ks.cend() && after->stamp < stamp;
       ++before, ++after) {
  }

  const double ratio =
      (stamp - before->stamp).to_sec() / (after->stamp - before->stamp).to_sec();

  return gtsam::interpolate(before->nav_state.pose(), after->nav_state.pose(),
                            ratio);
}

// ------------------------- Imu handler ------------------------- //

std::optional<Stamped<gtsam::NavState>> ImuHandler::register_imu(const Imu &imu) {
  std::scoped_lock lock(m_mutex);

  // If it's the first IMU measurement, nothing we can do with it yet
  if (m_buffer.empty()) {
    Duration zero = Duration::from_sec(0);
    auto imu_delta = ImuState{
        .delta = zero,
        .stamp = imu.stamp,
        .gyro = imu.gyro,
        .acc = imu.acc,
        .nav_state =
            gtsam::NavState(gtsam::Pose3::Identity(), gtsam::Velocity3::Zero()),
    };
    m_buffer.push_back(imu_delta);
    return std::nullopt;
  }

  // If it's too old, also return early
  if (imu.stamp < latest().stamp) {
    return std::nullopt;
  }

  // Finally, if it's good, integrate, save, and return the pose
  auto dt = imu.stamp - latest().stamp;

  auto nav_state = integrate(latest().nav_state, imu.gyro, imu.acc, m_bias,
                             m_params.preintegration->n_gravity, dt.to_sec());

  auto imu_delta = ImuState{
      .delta = dt,
      .stamp = imu.stamp,
      .gyro = imu.gyro,
      .acc = imu.acc,
      .nav_state = nav_state,
  };
  m_buffer.push_back(imu_delta);

  return StampedNavState{.stamp = latest().stamp, .data = latest().nav_state};
}

bool ImuHandler::ready_gravity_alignment() const noexcept {
  return m_buffer.size() >= m_params.initial_quota;
}

std::optional<std::pair<gtsam::Pose3, gtsam::imuBias::ConstantBias>>
ImuHandler::compute_gravity_alignment() noexcept {
  Eigen::Vector3d imu_gravity = Eigen::Vector3d::Zero();
  Eigen::Vector3d body_gyro = Eigen::Vector3d::Zero();

  // If there's not enough, try again later
  if (m_buffer.size() < m_params.initial_quota) {
    return {};
  }

  // Gather all IMU data up to the scan
  for (auto &imu : m_buffer) {
    imu_gravity -= imu.acc;
    body_gyro += imu.gyro;
  }
  const auto countDouble = static_cast<double>(m_buffer.size());
  imu_gravity /= countDouble;
  body_gyro /= countDouble;

  const gtsam::Pose3 world_T_imu = gtsam::Pose3::Identity();

  // Compute direction of gravity in identity frame
  Eigen::Vector3d imu_gravity_normed = imu_gravity.normalized() * 9.81;
  Eigen::Vector3d bias_accel = imu_gravity_normed - imu_gravity;
  m_params.preintegration->n_gravity =
      world_T_imu.rotation().unrotate(imu_gravity_normed);

  // // blenheim palace values
  // // from ground truth poses
  // m_params.preintegration->n_gravity = gtsam::Vector3(-0.066, -0.068, -0.995)
  // * 9.81; bias_accel = gtsam::Vector3(0.05, -0.009, -0.149); body_gyro =
  // gtsam::Vector3(0.005, 0.001, 0.);
  // // from FORM poses
  // m_params.preintegration->n_gravity = gtsam::Vector3(-0.027, -0.08, -0.996)
  // * 9.81; bias_accel = gtsam::Vector3(0.183, -0.285, -0.117);
  // body_gyro = gtsam::Vector3(0.005, 0.001, 0.);

  // short_experiment values
  // from FORM poses - gt/FORM results are similar
  // m_params.preintegration->n_gravity = gtsam::Vector3(0.01, 0.01, -1.0) * 9.81;
  // bias_accel = gtsam::Vector3(-0.308, -0.169, 0.067);
  // body_gyro = gtsam::Vector3(-0.019, -0.031, 0.005);

  // multi_campus/tuhh_night_09 values
  // from GT poses
  m_params.preintegration->n_gravity = gtsam::Vector3(0.084, 0.016, 0.996) * 9.81;
  bias_accel = gtsam::Vector3(-0.03, 0.093, 0.261);
  body_gyro = gtsam::Vector3(-0.001, 0., -0.008);

  const gtsam::imuBias::ConstantBias imu_bias(bias_accel, body_gyro);

  return std::make_pair(world_T_imu, imu_bias);
}

// TODO: Verify this function (I think it's correct)
void ImuHandler::update_from(const Stamp &stamp, const gtsam::NavState &nav_state,
                             const gtsam::imuBias::ConstantBias &bias) noexcept {
  std::scoped_lock lock(m_mutex);

  // Make sure our imu data is new enough
  if (m_buffer.empty() || latest().stamp < stamp) {
    std::cout << "ImuHandler::update_from: will drop all imu data" << std::endl;
    return;
  }

  // First drop all imu measurements, except the one directly before the stamp
  while (!m_buffer.empty() && (m_buffer.begin() + 1)->stamp < stamp) {
    m_buffer.pop_front();
  }

  // Make the one directly before the stamp occur AT the stamp
  // And fix the dt right after it
  // This essentially "adds" a measurement in at the stamp
  auto before = m_buffer.begin();
  auto after = before + 1;

  before->stamp = stamp;
  before->nav_state = nav_state;
  after->delta = after->stamp - stamp;
  m_bias = bias;

  // Finally, reintegrate all the IMU data
  for (auto imu_iter = m_buffer.begin() + 1; imu_iter != m_buffer.end();
       ++imu_iter) {
    imu_iter->nav_state =
        integrate((imu_iter - 1)->nav_state, imu_iter->gyro, imu_iter->acc, m_bias,
                  m_params.preintegration->n_gravity, imu_iter->delta.to_sec());
  }
}

// TODO: Verify this function
gtsam::PreintegratedCombinedMeasurements
ImuHandler::preintegrate(const Stamp &start, const Stamp &end) const noexcept {
  gtsam::PreintegratedCombinedMeasurements preintegrated(m_params.preintegration,
                                                         m_bias);

  //  TODO: Can remove start parameter, keeping it in as a sanity check
  // update_from should be called before this function to set the oldest IMU
  assert(!m_buffer.empty() &&
         "ImuHandler::preintegrate: no IMU data to preintegrate");
  assert(oldest().stamp == start &&
         "ImuHandler::preintegrate: start is not the oldest");
  assert(latest().stamp > end &&
         "ImuHandler::preintegrate: latest imu measurement is before end");

  // Skip to the correct time
  auto imu_iter = m_buffer.cbegin() + 1;

  // Integrate all IMU data up to the end
  // Don't check if we're at the end, asserts should handle that above
  while (imu_iter->stamp < end) {
    preintegrated.integrateMeasurement(imu_iter->acc, imu_iter->gyro,
                                       imu_iter->delta.to_sec());
    ++imu_iter;
  }

  // This should integrate the last IMU measurement that is
  // either equal or right after the end
  double dt = (end - (imu_iter - 1)->stamp).to_sec();
  if (dt >= 1e-8) {
    preintegrated.integrateMeasurement(imu_iter->acc, imu_iter->gyro, dt);
  }

  return preintegrated;
}

// TODO: Verify this function
const gtsam::NavState ImuHandler::at(const Stamp &stamp) const noexcept {
  auto before = m_buffer.cbegin();
  auto after = before + 1;
  for (;
       before != m_buffer.cend() && after != m_buffer.cend() && after->stamp < stamp;
       ++before, ++after) {
  }

  const double ratio =
      (stamp - before->stamp).to_sec() / (after->stamp - before->stamp).to_sec();

  auto pose =
      gtsam::interpolate(before->nav_state.pose(), after->nav_state.pose(), ratio);
  auto vel = gtsam::interpolate(before->nav_state.v(), after->nav_state.v(), ratio);

  return gtsam::NavState(pose, vel);
}

} // namespace form