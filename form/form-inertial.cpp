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
#include "form/utils.hpp"
#include <gtsam/navigation/NavState.h>
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
InertialEstimator::register_scan(const std::vector<Eigen::Vector3f> &scan,
                                 const Stamp &stamp) noexcept {
  constexpr auto SEQ = std::make_index_sequence<2>{};

  // TODO: Right now we're assuming all IMU measurements / LiDAR scans are coming in
  // order. Need to check other systems to see how it's handled robustly
  // TODO: Have to make sure stamp is within the IMU range (likely prior to here)
  // TODO: This isn't adding the keypoints to the map... needs to be handled
  // differently

  // Check if we're initialized, if not, do it!
  // TODO: This doesn't add things to the map!
  if (!m_constraints.initialized()) {
    auto kp = m_extractor.extract(scan, 0);
    // Try to align with gravity if we have enough IMU data
    auto aligned = m_imu.compute_gravity_alignment();
    if (!aligned.has_value()) {
      return kp;
    }
    // Initialize everything with correct pose
    m_constraints.initialize(aligned->first, aligned->second);
    return kp;
  }

  //
  // ############################ Feature Extraction ############################ //
  //
  // ------------------------------ Initialization ------------------------------ //
  // This needs to go first to get the scan index
  gtsam::NavState prediction = m_imu.at(stamp);
  auto [scan_idx, scan_constraints] =
      m_constraints.step(prediction.pose(), prediction.v());

  // ----------------------------- Extract Features ----------------------------- //
  auto keypoints = m_extractor.extract(scan, scan_idx);
  const auto num_keypoints =
      std::apply([](auto &...kps) { return (kps.size() + ...); }, keypoints);

  // Transform them into IMU frame
  tuple::for_each(keypoints, [&](auto &kps) {
    for (auto &kp : kps) {
      kp.transform_in_place(m_params.imu.imu_T_lidar);
    }
  });

  //
  // ############################### Optimization ############################### //
  //
  // ---------------------------- Add in IMU Factor ---------------------------- //
  m_constraints.add_imu_factor(m_imu.preintegrate(m_imu.oldest().stamp, stamp));

  // ---------------------------- Generate World Map ---------------------------- //
  const auto world_map = tuple::transform(m_keypoint_map, [&](auto &map) {
    return map.to_voxel_map(m_constraints.get_values(),
                            // make voxel size match the max matching distance
                            m_params.matcher.max_dist_matching);
  });

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

  // ------------------------ Full Nonlinear Optimization ------------------------ //
  new_values = m_constraints.optimize(false);
  m_constraints.update_values(new_values);

  //
  // ################################## Mapping ################################## //
  //
  // ---------------------------- Update IMU Handler ---------------------------- //
  auto bias = m_constraints.get_current_bias();
  auto nav_state = gtsam::NavState(m_constraints.get_current_pose(),
                                   m_constraints.get_current_velocity());
  m_imu.update_from(stamp, nav_state, bias);

  // ------------------------------ Map insertions ------------------------------ //
  tuple::for_seq(SEQ, [&](auto I) {
    std::get<I>(m_keypoint_map).insert_matches(std::get<I>(m_matcher).get_matches());
  });

  // ---------------------------- Keyscan Selection ---------------------------- //
  const auto connections = [&](ScanIndex i) {
    return m_constraints.num_recent_connections(i, m_keyscanner.oldest_rf());
  };
  auto marg_scans = m_keyscanner.step(scan_idx, num_keypoints, connections);

  // ----------------------------- Marginalization ----------------------------- //
  m_constraints.marginalize(marg_scans);
  tuple::for_each(m_keypoint_map, [&](auto &map) { map.remove(marg_scans); });

  return keypoints;
}

} // namespace form
