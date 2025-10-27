// MIT License

// Copyright (c) 2025 Easton Potokar, Taylor Pool, and Michael Kaess

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "form/form-inertial.hpp"
#include "form/inertial/imu.hpp"
#include "form/utils.hpp"
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/NavState.h>
#include <iostream>
#include <optional>

namespace form {

using gtsam::Pose3;
using gtsam::Velocity3;
using gtsam::symbol_shorthand::X;

InertialEstimator::InertialEstimator(
    const InertialEstimator::Params &params) noexcept
    : m_params(params), m_extractor(params.extraction, params.num_threads),
      m_constraints(params.constraints),
      m_matcher{Matcher<PlanarFeat>(params.matcher, params.num_threads),
                Matcher<PointFeat>(params.matcher, params.num_threads)},
      m_keyscanner(params.scans), m_keypoint_map{KeypointMap<PlanarFeat>(params.map),
                                                 KeypointMap<PointFeat>(params.map)},
      m_imu(params.imu) {}

std::optional<Stamped<gtsam::NavState>>
InertialEstimator::register_imu(const Imu &imu) {
  return m_imu.register_imu(imu);
}

std::tuple<std::vector<PlanarFeat>, std::vector<PointFeat>>
InertialEstimator::register_single_scan(const PointCloud &scan) noexcept {
  constexpr auto SEQ = std::make_index_sequence<2>{};
  Timer timer;

  //
  // ############################ Feature Extraction ############################ //
  //
  // ------------------------------ Initialization ------------------------------ //
  // Try to get prediction from IMU if not initialized yet
  gtsam::Pose3 pose;
  gtsam::Velocity3 vel;
  gtsam::imuBias::ConstantBias bias;
  if (!m_constraints.initialized()) {
    auto aligned = m_imu.compute_gravity_alignment();
    std::tie(pose, bias) = aligned.value();
  } else {
    gtsam::NavState nav_state = m_imu.at(scan.stamp);
    pose = nav_state.pose();
    vel = nav_state.v();
  }

  // This needs to go first to get the scan index
  // const auto prediction = m_constraints.predict_next();
  auto [scan_idx, scan_constraints] =
      !m_constraints.initialized()
          ? m_constraints.initialize(pose, bias)
          // : m_constraints.step(prediction, m_constraints.get_current_velocity());
          : m_constraints.step(pose, m_constraints.get_current_velocity());
  timer.elapsed("Initialization");

  // ----------------------------- Extract Features ----------------------------- //
  auto keypoints = m_extractor.extract(scan.data, scan_idx);
  auto keypoints_lidar = keypoints; // for returning in lidar frame
  const auto num_keypoints =
      std::apply([](auto &...kps) { return (kps.size() + ...); }, keypoints);

  // Transform them into IMU frame
  tuple::for_each(keypoints, [&](auto &kps) {
    for (auto &kp : kps) {
      kp.transform_in_place(m_params.imu.imu_T_lidar);
    }
  });
  timer.elapsed("Feature Extraction");

  //
  // ############################### Optimization ############################### //
  //
  // ---------------------------- Add in IMU Factor ---------------------------- //
  if (scan_idx != 0) {
    m_constraints.add_imu_factor(
        m_imu.preintegrate(m_imu.oldest().stamp, scan.stamp));
  }
  timer.elapsed("Add IMU Factor");

  // ---------------------------- Generate World Map ---------------------------- //
  const auto world_map = tuple::transform(m_keypoint_map, [&](auto &map) {
    return map.to_voxel_map(m_constraints.get_values(),
                            // make voxel size match the max matching distance
                            m_params.matcher.max_dist_matching);
  });
  timer.elapsed("Generate World Map");

  // ICP loop
  gtsam::Values new_values;
  auto estimates = [&](size_t i) { return m_constraints.get_pose(i); };
  for (size_t idx = 0; idx < m_params.matcher.max_num_rematches; ++idx) {
    auto before = m_constraints.get_current_pose();

    // -------------------------------- Matching -------------------------------- //
    // Match each type of feature
    tuple::for_seq(SEQ, [&](auto I) {
      std::get<I>(m_matcher).template match<I>(std::get<I>(world_map),
                                               std::get<I>(keypoints), estimates,
                                               scan_constraints);
    });

    // ---------------------- Semi-Linearized Optimization ---------------------- //
    new_values = m_constraints.optimize(true);
    const auto after = new_values.at<Pose3>(X(scan_idx));
    const auto diff = before.localCoordinates(after).norm();
    if (diff < m_params.matcher.new_pose_threshold) {
      break;
    }
    m_constraints.update_current_pose(after);
  }
  timer.elapsed("ICP Loop Time");

  // ------------------------ Full Nonlinear Optimization ------------------------ //
  new_values = m_constraints.optimize(false);
  m_constraints.update_values(new_values);
  timer.elapsed("Full Nonlinear Optimization Time");

  //
  // ################################## Mapping ################################## //
  //
  // ---------------------------- Update IMU Handler ---------------------------- //
  auto nav_state = gtsam::NavState(m_constraints.get_current_pose(),
                                   m_constraints.get_current_velocity());
  m_imu.update_from(scan.stamp, nav_state, m_constraints.get_current_bias());
  timer.elapsed("Update IMU Handler");

  // ------------------------------ Map insertions ------------------------------ //
  tuple::for_seq(SEQ, [&](auto I) {
    std::get<I>(m_keypoint_map).insert_matches(std::get<I>(m_matcher).get_matches());
  });
  timer.elapsed("Map Insertions");

  // ---------------------------- Keyscan Selection ---------------------------- //
  const auto connections = [&](ScanIndex i) {
    return m_constraints.num_recent_connections(i, m_keyscanner.oldest_rf());
  };
  auto marg_scans = m_keyscanner.step(scan_idx, num_keypoints, connections);
  timer.elapsed("Keyscan Selection");

  // ----------------------------- Marginalization ----------------------------- //
  m_constraints.marginalize(marg_scans);
  tuple::for_each(m_keypoint_map, [&](auto &map) { map.remove(marg_scans); });
  timer.elapsed("Marginalization");

  // Save pose
  m_estimates.push_back(
      Stamped<Pose3>{.stamp = scan.stamp, .data = m_constraints.get_current_pose()});

  return keypoints_lidar;
}

std::optional<std::tuple<std::vector<PlanarFeat>, std::vector<PointFeat>>>
InertialEstimator::register_scan(const std::vector<PointXYZf> &scan,
                                 const Stamp &stamp) noexcept {

  // Check if we should save the scan
  if (m_scan.empty() || m_scan.back().stamp < stamp) {
    m_scan.push_back(PointCloud{.stamp = stamp, .data = scan});
  }

  std::optional<std::tuple<std::vector<PlanarFeat>, std::vector<PointFeat>>> result =
      std::nullopt;
  while (
      // Have scans
      !m_scan.empty()
      // Have IMU data
      && !m_imu.empty()
      // IMU data is newer than scan
      && m_scan.front().stamp < m_imu.latest().stamp
      // Either initialized or ready to initialize
      && (m_constraints.initialized() || m_imu.ready_gravity_alignment())
      //
  ) {
    result = register_single_scan(m_scan.front());
    m_scan.pop_front();
  }

  return result;
}

} // namespace form
