#pragma once

#include "form/inertial/timing.hpp"

#include <Eigen/Core>
#include <deque>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/NavState.h>
#include <mutex>
#include <optional>

namespace form {

// TODO: Move some of these types to a central location
template <typename T> struct Stamped {
  Stamp stamp;
  T data;

  [[nodiscard]] constexpr operator bool() const noexcept { return bool(stamp); }
};

using StampedNavState = Stamped<gtsam::NavState>;

struct Imu {
  Stamp stamp;
  Eigen::Vector3d gyro;
  Eigen::Vector3d acc;
};

struct ImuState {
  Duration delta;
  Stamp stamp;
  Eigen::Vector3d gyro;
  Eigen::Vector3d acc;
  gtsam::NavState nav_state;
};

[[nodiscard]] gtsam::Pose3 interpolate(const std::deque<ImuState> &w_T_ks,
                                       const Stamp &stamp) noexcept;

class ImuHandler {
public:
  struct Params {
    /// @brief Initial required number of IMU measurements to compute gravity
    /// alignment
    size_t initial_quota = 400;

    /// @brief Preintegration parameters
    boost::shared_ptr<gtsam::PreintegrationCombinedParams> preintegration =
        gtsam::PreintegrationCombinedParams::MakeSharedU();

    /// @brief Transform from IMU to LiDAR
    gtsam::Pose3 imu_T_lidar = gtsam::Pose3::Identity();
  };

private:
  Params m_params;

  // data
  std::deque<ImuState> m_buffer;       // holds raw imu measurements
  gtsam::imuBias::ConstantBias m_bias; // bias estimate

  std::mutex m_mutex;

public:
  ImuHandler()
      : m_params(Params{.preintegration =
                            gtsam::PreintegrationCombinedParams::MakeSharedU()}) {};
  ImuHandler(const Params &params) : m_params(params) {}

  // ------------------------- Doers ------------------------- //
  /// @brief Add IMU data to the buffer and integrate it
  std::optional<Stamped<gtsam::NavState>> register_imu(const Imu &imu);

  /// @brief Get initial pose and bias aligned with gravity
  std::optional<std::pair<gtsam::Pose3, gtsam::imuBias::ConstantBias>>
  compute_gravity_alignment() noexcept;

  // Reset all integration to begin at stamp, and re-integrate
  void update_from(const Stamp &stamp, const gtsam::NavState &nav_state,
                   const gtsam::imuBias::ConstantBias &bias) noexcept;

  // Compute preintegration
  gtsam::PreintegratedCombinedMeasurements
  preintegrate(const Stamp &start, const Stamp &end) const noexcept;

  // Dewarp a scan
  // template <typename Point>
  // void dewarp(PointCloud<Point> &ks_points,
  //             const gtsam::Pose3 &w_T_k) const noexcept {
  //   const gtsam::Pose3 k_T_w = w_T_k.inverse();

  //   std::for_each(ks_points.points.begin(), ks_points.points.end(),
  //                 [&](Point &ki_point) noexcept {
  //                   const auto w_T_ki =
  //                       interpolate(m_buffer, ki_point.timeOffset +
  //                       ks_points.stamp);
  //                   ki_point.transform_in_place(k_T_w.transformPoseFrom(w_T_ki));
  //                 });
  // }

  // ------------------------- Getters ------------------------- //
  bool ready_gravity_alignment() const noexcept;
  const gtsam::imuBias::ConstantBias &bias() const noexcept { return m_bias; }
  const ImuState &oldest() const noexcept { return m_buffer.front(); }
  const ImuState &latest() const noexcept { return m_buffer.back(); }
  const ImuState &get(size_t i) const noexcept { return m_buffer[i]; }
  bool empty() const noexcept { return m_buffer.empty(); }
  size_t size() const noexcept { return m_buffer.size(); }
  const gtsam::NavState at(const Stamp &stamp) const noexcept;
};

} // namespace form