#include "evalio/convert/base.h"
#include "evalio/convert/eigen.h"
#include "evalio/convert/gtsam.h"
#include "evalio/macros.h"
#include "evalio/pipeline.h"
#include "evalio/types.h"

#include "form/feature/extraction.hpp"
#include "form/feature/features.hpp"
#include "form/form-inertial.hpp"
#include "form/form.hpp"
#include "form/inertial/imu.hpp"
#include "form/inertial/init_factor.hpp"
#include "form/utils.hpp"

#include <cstdio>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <memory>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
namespace ev = evalio;

// ------------------------- Helpers ------------------------- //
namespace evalio {

// point conversions
template <> inline Point convert(const form::PointFeat &in) {
  return {
      .x = in.x,
      .y = in.y,
      .z = in.z,
      .intensity = 0.0,
      .t = Duration::from_sec(0),
      .row = 0,
      .col = static_cast<uint16_t>(in.scan),
  };
}

template <> inline Point convert(const form::PlanarFeat &in) {
  return {
      .x = in.x,
      .y = in.y,
      .z = in.z,
      .intensity = 0.0,
      .t = Duration::from_sec(0),
      .row = 0,
      .col = static_cast<uint16_t>(in.scan),
  };
}

template <> inline form::PointXYZf convert(const Point &in) {
  return {
      static_cast<float>(in.x),
      static_cast<float>(in.y),
      static_cast<float>(in.z),
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

// ------------------------- Pipeline ------------------------- //
class FORMDev : public ev::Pipeline {
public:
  FORMDev() : estimator_(), params_() {}

  // Info
  static std::string version() { return "0.2.0"; } // x-release-please-version

  static std::string name() { return "form-dev"; }

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
    (double,           min_dist_map, 0.1, params_.map.min_dist_map),
    // misc
    (int, num_threads, 0, params_.num_threads)
  );
  // clang-format on

  // Getters
  const std::map<std::string, std::vector<ev::Point>> map() override {
    const auto planar = std::get<0>(estimator_.m_keypoint_map)
                            .to_vector(estimator_.m_constraints.get_values());
    const auto point = std::get<1>(estimator_.m_keypoint_map)
                           .to_vector(estimator_.m_constraints.get_values());

    return ev::make_map("planar", planar, "point", point);
  }

  // Setters
  void set_imu_params(ev::ImuParams params) override {}

  void set_lidar_params(ev::LidarParams params) override {
    params_.extraction.min_norm_squared = params.min_range * params.min_range;
    params_.extraction.max_norm_squared = params.max_range * params.max_range;
    params_.extraction.num_columns = params.num_columns;
    params_.extraction.num_rows = params.num_rows;
  }

  void set_imu_T_lidar(ev::SE3 T) override {
    lidar_T_imu_ = ev::convert<gtsam::Pose3>(T).inverse();
  }

  // Doers
  void initialize() override { estimator_ = form::Estimator(params_); }

  void add_imu(ev::ImuMeasurement mm) override {}

  void add_lidar(ev::LidarMeasurement mm) override {
    const auto scan = ev::convert_iter<std::vector<form::PointXYZf>>(mm.points);

    auto [planar_kp, point_kp] = estimator_.register_scan(scan);

    this->save(mm.stamp, estimator_.current_lidar_estimate() * lidar_T_imu_);
    this->save(mm.stamp, "planar", planar_kp, "point", point_kp);
  }

private:
  form::Estimator estimator_;
  form::Estimator::Params params_;

  gtsam::Pose3 lidar_T_imu_ = gtsam::Pose3::Identity();
};

class FORMInertialDev : public ev::Pipeline {
public:
  FORMInertialDev() : estimator_(), params_() {}

  // Info
  static std::string version() { return "0.2.0"; } // x-release-please-version

  static std::string name() { return "form-inertial-dev"; }

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
    (double,           min_dist_map, 0.1, params_.map.min_dist_map),
    // misc
    (int, num_threads, 0, params_.num_threads)
  );
  // clang-format on

  // Getters
  const std::map<std::string, std::vector<ev::Point>> map() override {
    const auto planar = std::get<0>(estimator_->m_keypoint_map)
                            .to_vector(estimator_->m_constraints.get_values());
    const auto point = std::get<1>(estimator_->m_keypoint_map)
                           .to_vector(estimator_->m_constraints.get_values());

    return ev::make_map("planar", planar, "point", point);
  }

  // Setters
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

  void set_lidar_params(ev::LidarParams params) override {
    params_.extraction.min_norm_squared = params.min_range * params.min_range;
    params_.extraction.max_norm_squared = params.max_range * params.max_range;
    params_.extraction.num_columns = params.num_columns;
    params_.extraction.num_rows = params.num_rows;
    delta_time_ = params.delta_time();
  }

  void set_imu_T_lidar(ev::SE3 T) override {
    params_.imu.imu_T_lidar = ev::convert<gtsam::Pose3>(T);
  }

  // Doers
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

private:
  // must be a shared ptr due to mutex in imu class
  std::shared_ptr<form::InertialEstimator> estimator_;
  form::InertialEstimator::Params params_;
  ev::SE3 last_integrated_imu_ = ev::SE3::identity();

  // helper params
  ev::Duration delta_time_;
};

NB_MODULE(_core, m) {
  m.doc() = "Custom evalio pipeline example";

  // gtsam bindings if we want them
  nb::module_ gtsam = nb::module_::import_("gtsam");
  nb::module_ eval = nb::module_::import_("evalio");

  // Only have to override the static methods here
  // All the others will be automatically inherited from the base class
  nb::class_<FORMDev, evalio::Pipeline>(m, "FORMDev")
      .def(nb::init<>())
      .def_static("name", &FORMDev::name)
      .def_static("url", &FORMDev::url)
      .def_static("default_params", &FORMDev::default_params);

  nb::class_<FORMInertialDev, ev::Pipeline>(m, "FORMInertialDev")
      .def(nb::init<>())
      .def_static("name", &FORMInertialDev::name)
      .def_static("url", &FORMInertialDev::url)
      .def_static("default_params", &FORMInertialDev::default_params)
      .def("get_last_integrated_imu", &FORMInertialDev::get_last_integrated_imu)
      .def("current_imu_bias", &FORMInertialDev::current_imu_bias);

  nb::class_<form::GravityBiasPreintFactor, gtsam::NoiseModelFactor>(
      m, "GravityBiasPreintFactor")
      .def(nb::init<const gtsam::PreintegratedCombinedMeasurements &,
                    const gtsam::Pose3 &, const gtsam::Pose3 &,
                    const gtsam::Velocity3 &, gtsam::Key, gtsam::Key>(),
           nb::arg("preint_imu"), nb::arg("x0"), nb::arg("x1"), nb::arg("v0"),
           nb::arg("bias_key"), nb::arg("gravity_key"));
  // .def("evaluateError", &form::GravityBiasPreintFactor::evaluateError,
  //      nb::arg("bias"), nb::arg("gravity"), nb::arg("H1") = nullptr,
  //      nb::arg("H2") = nullptr);

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
