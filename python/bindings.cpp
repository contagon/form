#include "evalio/convert/base.h"
#include "evalio/convert/eigen.h"
#include "evalio/macros.h"
#include "evalio/pipeline.h"
#include "evalio/types.h"

#include "form/feature/extraction.hpp"
#include "form/feature/features.hpp"
#include "form/form-inertial.hpp"
#include "form/form.hpp"
#include "form/inertial/imu.hpp"
#include "form/utils.hpp"

#include <cstdio>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <memory>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <utility>

namespace nb = nanobind;
namespace ev = evalio;

// ------------------------- Helpers ------------------------- //
namespace evalio {
// Poses
template <> gtsam::Pose3 convert(const ev::SE3 &pose) {
  return gtsam::Pose3(gtsam::Rot3(pose.rot.toEigen()), gtsam::Point3(pose.trans));
}

template <> ev::SE3 convert(const gtsam::Pose3 &pose) {
  return ev::SE3(ev::SO3::fromEigen(pose.rotation().toQuaternion()),
                 pose.translation());
}

// Points
template <> form::PointXYZf convert(const ev::Point &point) {
  return form::PointXYZf(point.x, point.y, point.z);
}

template <> ev::Point convert(const form::PointFeat &point) {
  return {
      .x = point.x,
      .y = point.y,
      .z = point.z,
      .intensity = 0.0,
      .t = ev::Duration::from_sec(0),
      .row = 0,
      .col = static_cast<uint16_t>(point.scan),
  };
}

template <> ev::Point convert(const form::PlanarFeat &point) {
  return {
      .x = point.x,
      .y = point.y,
      .z = point.z,
      .intensity = 0.0,
      .t = ev::Duration::from_sec(0),
      .row = 0,
      .col = static_cast<uint16_t>(point.scan),
  };
}

// imu
template <> form::Imu convert(const ev::ImuMeasurement &mm) {
  return form::Imu{
      .stamp = form::Stamp{.sec = mm.stamp.sec, .nsec = mm.stamp.nsec},
      .gyro = mm.gyro,
      .acc = mm.accel,
  };
}

// stamps
template <> form::Stamp convert(const ev::Stamp &stamp) {
  return form::Stamp{.sec = stamp.sec, .nsec = stamp.nsec};
}

template <> ev::Stamp convert(const form::Stamp &stamp) {
  return ev::Stamp{.sec = stamp.sec, .nsec = stamp.nsec};
}

} // namespace evalio

// ------------------------- Pipelines ------------------------- //
class FORM : public evalio::Pipeline {
public:
  FORM() : evalio::Pipeline(), params_(), estimator_() {}

  form::Estimator estimator_;
  form::Estimator::Params params_;

  // helper params
  gtsam::Pose3 lidar_T_imu_ = gtsam::Pose3::Identity();
  evalio::Duration delta_time_;

  // ------------------------- Info ------------------------- //
  static std::string name() { return "form"; }

  static std::string url() { return "https://github.com/rpl-cmu/form"; }

  // clang-format off
  EVALIO_SETUP_PARAMS(
    // FEATURE EXTRACTION
    (int,         neighbor_points,   5, params_.extraction.neighbor_points),
    (int,             num_sectors,   6, params_.extraction.num_sectors),
    (double,     planar_threshold, 1.0, params_.extraction.planar_threshold),
    (int, planar_feats_per_sector,  50, params_.extraction.planar_feats_per_sector),
    (int,  point_feats_per_sector,   3, params_.extraction.point_feats_per_sector),
    (double,               radius, 1.0, params_.extraction.radius),
    (int,              min_points,   5, params_.extraction.min_points),
    // OPTIMIZATION
    (double,  max_dist_matching,   0.8, params_.matcher.max_dist_matching),
    (double, new_pose_threshold,  1e-4, params_.matcher.new_pose_threshold),
    (int,     max_num_rematches,    30, params_.matcher.max_num_rematches),
    (bool,    disable_smoothing, false, params_.constraints.disable_smoothing),
    // MAPPING
    (int,         max_num_keyscans,  50, params_.scans.max_num_keyscans),
    (int,     max_num_recent_scans,  10, params_.scans.max_num_recent_scans),
    (int, max_steps_unused_keyscan,  10, params_.scans.max_steps_unused_keyscan),
    (double,    keyscan_match_ratio, 0.1, params_.scans.keyscan_match_ratio),
    (double,           max_dist_map, 0.1, params_.map.min_dist_map),
    // misc
    (int, num_threads, 0, params_.num_threads)
  );
  // clang-format on

  // ------------------------- Getters ------------------------- //
  const std::map<std::string, std::vector<evalio::Point>> map() override {
    const auto [planar, point] =
        form::tuple::transform(estimator_.m_keypoint_map, [&](auto &map) {
          return map.to_vector(estimator_.m_constraints.get_values());
        });

    return evalio::make_map("planar", planar, "point", point);
  }

  // ------------------------- Setters ------------------------- //
  // Set the IMU parameters
  void set_imu_params(evalio::ImuParams params) override {}

  // Set the LiDAR parameters
  void set_lidar_params(evalio::LidarParams params) override {
    params_.extraction.min_norm_squared = params.min_range * params.min_range;
    params_.extraction.max_norm_squared = params.max_range * params.max_range;
    params_.extraction.num_columns = params.num_columns;
    params_.extraction.num_rows = params.num_rows;
    delta_time_ = params.delta_time();
  }

  // Set the transformation from IMU to LiDAR
  void set_imu_T_lidar(evalio::SE3 T) override {
    lidar_T_imu_ = ev::convert<gtsam::Pose3>(T).inverse();
  }

  // ------------------------- Doers ------------------------- //
  // Initialize the pipeline
  void initialize() override { estimator_ = form::Estimator(params_); }

  // Add an IMU measurement
  void add_imu(evalio::ImuMeasurement mm) override {}

  // Add a LiDAR measurement
  void add_lidar(evalio::LidarMeasurement mm) override {
    // convert to evalio
    auto scan = ev::convert_iter<std::vector<form::PointXYZf>>(mm.points);

    // run the estimator & save results
    auto [planar_kp, point_kp] = estimator_.register_scan(scan);
    this->save(mm.stamp, estimator_.current_lidar_estimate() * lidar_T_imu_);

    // extract the keypoints
    this->save(mm.stamp, "planar", planar_kp, "point", point_kp);
    std::map<std::string, std::vector<evalio::Point>> points = {{"planar", {}},
                                                                {"point", {}}};
  }
};

class FORMInertial : public ev::Pipeline {
public:
  FORMInertial() : ev::Pipeline(), params_() {}

  // must be a shared ptr due to mutex in imu class
  std::shared_ptr<form::InertialEstimator> estimator_;
  form::InertialEstimator::Params params_;
  ev::SE3 last_integrated_imu_ = ev::SE3::identity();

  // helper params
  ev::Duration delta_time_;

  // ------------------------- Info ------------------------- //
  static std::string name() { return "form-inertial"; }

  static std::string url() { return "https://github.com/rpl-cmu/form"; }

  // clang-format off
  EVALIO_SETUP_PARAMS(
    // FEATURE EXTRACTION
    (int,         neighbor_points,   5, params_.extraction.neighbor_points),
    (int,             num_sectors,   6, params_.extraction.num_sectors),
    (double,     planar_threshold, 1.0, params_.extraction.planar_threshold),
    (int, planar_feats_per_sector,  50, params_.extraction.planar_feats_per_sector),
    (int,  point_feats_per_sector,   3, params_.extraction.point_feats_per_sector),
    (double,               radius, 1.0, params_.extraction.radius),
    (int,              min_points,   5, params_.extraction.min_points),
    // OPTIMIZATION
    (double,  max_dist_matching,   0.8, params_.matcher.max_dist_matching),
    (double, new_pose_threshold,  1e-4, params_.matcher.new_pose_threshold),
    (int,     max_num_rematches,    30, params_.matcher.max_num_rematches),
    (bool,    disable_smoothing, false, params_.constraints.disable_smoothing),
    (double,       planar_sigma,   0.1, params_.constraints.planar_constraint_sigma),
    (double,        prior_scale,   1.0, params_.constraints.prior_scale),
    // MAPPING
    (int,         max_num_keyscans,   50, params_.scans.max_num_keyscans),
    (int,     max_num_recent_scans,   10, params_.scans.max_num_recent_scans),
    (int, max_steps_unused_keyscan,   10, params_.scans.max_steps_unused_keyscan),
    (double,    keyscan_match_ratio, 0.1, params_.scans.keyscan_match_ratio),
    (double,           max_dist_map, 0.1, params_.map.min_dist_map),
    // misc
    (int, num_threads,     0, params_.num_threads),
  );
  // clang-format on

  // ------------------------- Getters ------------------------- //
  // Returns the current submap of the environment
  const std::map<std::string, std::vector<ev::Point>> map() override {
    const auto [planar, point] =
        form::tuple::transform(estimator_->m_keypoint_map, [&](auto &map) {
          return map.to_vector(estimator_->m_constraints.get_values());
        });

    return evalio::make_map("planar", planar, "point", point);
  }

  // ------------------------- Setters ------------------------- //
  // Set the IMU parameters
  void set_imu_params(ev::ImuParams params) override {
    auto cov3 = [](double std) { return std * std * Eigen::Matrix3d::Identity(); };
    auto cov6 = [](double std) {
      return std * std * Eigen::Matrix<double, 6, 6>::Identity();
    };
    params_.imu.preintegration->gyroscopeCovariance = cov3(params.gyro);
    params_.imu.preintegration->accelerometerCovariance = cov3(params.accel);
    params_.imu.preintegration->biasOmegaCovariance = cov3(params.gyro_bias);
    params_.imu.preintegration->biasAccCovariance = cov3(params.accel_bias);
    params_.imu.preintegration->integrationCovariance = cov3(params.integration);
    params_.imu.preintegration->biasAccOmegaInt = cov6(params.bias_init);
  }

  // Set the LiDAR parameters
  void set_lidar_params(ev::LidarParams params) override {
    params_.extraction.min_norm_squared = params.min_range * params.min_range;
    params_.extraction.max_norm_squared = params.max_range * params.max_range;
    params_.extraction.num_columns = params.num_columns;
    params_.extraction.num_rows = params.num_rows;
    delta_time_ = params.delta_time();
  }

  // Set the transformation from IMU to LiDAR
  void set_imu_T_lidar(ev::SE3 T) override {
    params_.imu.imu_T_lidar = ev::convert<gtsam::Pose3>(T);
  }

  // ------------------------- Doers ------------------------- //
  // Initialize the pipeline
  void initialize() override {
    // Make sure prior scale is set
    params_.constraints.set_scale();
    estimator_ = std::make_shared<form::InertialEstimator>(params_);
  }

  // Add an IMU measurement
  void add_imu(ev::ImuMeasurement mm) override {
    auto result = estimator_->register_imu(ev::convert<form::Imu>(mm));
    if (result.has_value()) {
      last_integrated_imu_ = ev::convert<ev::SE3>(result.value().data.pose());
    }
  }

  ev::SE3 get_last_integrated_imu() { return last_integrated_imu_; }

  Eigen::Matrix<double, 6, 1> current_imu_bias() {
    return estimator_->m_constraints.get_current_bias().vector();
  }

  // Add a LiDAR measurement
  void add_lidar(ev::LidarMeasurement mm) override {
    // convert to evalio
    auto scan = ev::convert_iter<std::vector<form::PointXYZf>>(mm.points);
    // send a stamp that is at the end of the scan
    auto end = ev::convert<form::Stamp>(mm.stamp + delta_time_);

    // Add in the scan
    estimator_->register_scan(scan, end);

    // Process any pending scans
    for (const auto &[stamp, pose, planar_kp, point_kp] :
         estimator_->process_pending()) {
      auto ev_stamp = ev::convert<ev::Stamp>(stamp) - delta_time_;
      this->save(ev_stamp, "planar", planar_kp, "point", point_kp);
      this->save(ev_stamp, pose);
    }
  }
};

NB_MODULE(_core, m) {
  m.doc() = "Custom evalio pipeline example";

  nb::module_ eval = nb::module_::import_("evalio");

  // Only have to override the static methods here
  // All the others will be automatically inherited from the base class
  nb::class_<FORM, evalio::Pipeline>(m, "FORM")
      .def(nb::init<>())
      .def_static("name", &FORM::name)
      .def_static("url", &FORM::url)
      .def_static("default_params", &FORM::default_params);

  nb::class_<FORMInertial, ev::Pipeline>(m, "FORMInertial")
      .def(nb::init<>())
      .def_static("name", &FORMInertial::name)
      .def_static("url", &FORMInertial::url)
      .def_static("default_params", &FORMInertial::default_params)
      .def("get_last_integrated_imu", &FORMInertial::get_last_integrated_imu)
      .def("current_imu_bias", &FORMInertial::current_imu_bias);

  // Expose extraction methods too
  nb::class_<form::FeatureExtractor::Params>(m, "KeypointExtractionParams")
      .def(nb::init<>())
      .def_rw("neighbor_points", &form::FeatureExtractor::Params::neighbor_points)
      .def_rw("num_sectors", &form::FeatureExtractor::Params::num_sectors)
      .def_rw("planar_feats_per_sector",
              &form::FeatureExtractor::Params::planar_feats_per_sector)
      .def_rw("planar_threshold", &form::FeatureExtractor::Params::planar_threshold)
      .def_rw("point_feats_per_sector",
              &form::FeatureExtractor::Params::point_feats_per_sector)
      // Parameters for normal estimation
      .def_rw("radius", &form::FeatureExtractor::Params::radius)
      .def_rw("min_points", &form::FeatureExtractor::Params::min_points)
      // Based on LiDAR info
      .def_rw("min_norm_squared", &form::FeatureExtractor::Params::min_norm_squared)
      .def_rw("max_norm_squared", &form::FeatureExtractor::Params::max_norm_squared)
      .def_rw("num_rows", &form::FeatureExtractor::Params::num_rows)
      .def_rw("num_columns", &form::FeatureExtractor::Params::num_columns);

  m.def("extract_keypoints", [](const std::vector<Eigen::Vector3d> &points,
                                const form::FeatureExtractor::Params &params,
                                ev::LidarParams &lidar_params) {
    // Convert to form points
    std::vector<form::PointXYZf> points_form;
    for (const auto &point : points) {
      points_form.emplace_back(point.x(), point.y(), point.z());
    }

    // Call the keypoint extraction function from the Tclio class
    auto [planar_keypoints, point_keypoints] =
        form::FeatureExtractor(params, 0).extract(points_form, 0);

    // return a tuple of (planar_points, normals, point_points)
    std::vector<Eigen::Vector3d> planar_points;
    std::vector<Eigen::Vector3d> normals;
    std::vector<Eigen::Vector3d> point_points;
    for (const auto &keypoint : planar_keypoints) {
      planar_points.emplace_back(keypoint.x, keypoint.y, keypoint.z);
      normals.emplace_back(keypoint.nx, keypoint.ny, keypoint.nz);
    }
    for (const auto &keypoint : point_keypoints) {
      point_points.emplace_back(keypoint.x, keypoint.y, keypoint.z);
    }

    return std::make_tuple(planar_points, normals, point_points);
  });
}
