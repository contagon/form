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

// ------------------------- Helpers ------------------------- //
gtsam::Pose3 pose_to_gtsam(const evalio::SE3 &pose) {
  return gtsam::Pose3(gtsam::Rot3(pose.rot.toEigen()), gtsam::Point3(pose.trans));
}

evalio::SE3 pose_to_evalio(const gtsam::Pose3 &pose) {
  return evalio::SE3(evalio::SO3::fromEigen(pose.rotation().toQuaternion()),
                     pose.translation());
}

template <typename Point> evalio::Point point_to_evalio(const Point &point) {
  return {
      .x = point.x,
      .y = point.y,
      .z = point.z,
      .intensity = 0.0,
      .t = evalio::Duration::from_sec(0),
      .row = 0,
      .col = static_cast<uint16_t>(point.scan),
  };
}

form::PointXYZf point_to_form(const evalio::Point &point) {
  return form::PointXYZf(point.x, point.y, point.z);
}

form::Imu imu_to_form(const evalio::ImuMeasurement &mm) {
  return form::Imu{
      .stamp = form::Stamp{.sec = mm.stamp.sec, .nsec = mm.stamp.nsec},
      .gyro = mm.gyro,
      .acc = mm.accel,
  };
}

// ------------------------- Pipeline ------------------------- //
class FORM : public evalio::Pipeline {
public:
  FORM() : evalio::Pipeline(), params_() {}

  // must be a shared ptr due to mutex in imu class
  std::shared_ptr<form::InertialEstimator> inertial_estimator_;
  std::shared_ptr<form::Estimator> estimator_;
  form::InertialEstimator::Params params_;
  bool fuse_imu_ = false;

  // helper params
  evalio::Duration delta_time_;
  evalio::SE3 current_pose = evalio::SE3::identity();

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
    (int,         max_num_keyscans,   50, params_.scans.max_num_keyscans),
    (int,     max_num_recent_scans,   10, params_.scans.max_num_recent_scans),
    (int, max_steps_unused_keyscan,   10, params_.scans.max_steps_unused_keyscan),
    (double,    keyscan_match_ratio, 0.1, params_.scans.keyscan_match_ratio),
    (double,           max_dist_map, 0.1, params_.map.min_dist_map),
    // misc
    (int, num_threads,     0, params_.num_threads),
    // helper to know which class to send data to
    (bool,   fuse_imu, false, fuse_imu_)
  );
  // clang-format on

  // ------------------------- Getters ------------------------- //
  // Returns the most recent pose estimate
  const evalio::SE3 pose() override { return current_pose; }

  // Returns the current submap of the environment
  const std::map<std::string, std::vector<evalio::Point>> map() override {
    using namespace form;
    const auto values = fuse_imu_ ? inertial_estimator_->m_constraints.get_values()
                                  : estimator_->m_constraints.get_values();

    // Get the points in the global frame from the keypoint map
    std::tuple<std::vector<PlanarFeat>, std::vector<PointFeat>> form_points;
    if (fuse_imu_) {
      form_points =
          tuple::transform(inertial_estimator_->m_keypoint_map,
                           [&](const auto &map) { return map.to_vector(values); });
    } else {
      form_points =
          tuple::transform(estimator_->m_keypoint_map,
                           [&](const auto &map) { return map.to_vector(values); });
    }

    // Convert to evalio format
    std::tuple<std::string, std::string> map_names = {"planar", "point"};
    std::map<std::string, std::vector<evalio::Point>> points;

    form::tuple::for_seq(std::make_index_sequence<2>{}, [&](auto I) {
      const auto name = std::get<I>(map_names);
      const auto &f = std::get<I>(form_points);

      points.insert({name, {}});
      auto &vec = points[name];
      vec.reserve(f.size());

      for (const auto &point : f) {
        vec.push_back(point_to_evalio(point));
      }
    });

    return points;
  }

  // ------------------------- Setters ------------------------- //
  // Set the IMU parameters
  void set_imu_params(evalio::ImuParams params) override {
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
  void set_lidar_params(evalio::LidarParams params) override {
    params_.extraction.min_norm_squared = params.min_range * params.min_range;
    params_.extraction.max_norm_squared = params.max_range * params.max_range;
    params_.extraction.num_columns = params.num_columns;
    params_.extraction.num_rows = params.num_rows;
    delta_time_ = params.delta_time();
  }

  // Set the transformation from IMU to LiDAR
  void set_imu_T_lidar(evalio::SE3 T) override {
    params_.imu.imu_T_lidar = pose_to_gtsam(T);
  }

  // ------------------------- Doers ------------------------- //
  // Initialize the pipeline
  void initialize() override {
    inertial_estimator_ = std::make_shared<form::InertialEstimator>(params_);

    // base estimator params are always a subset of inertial params
    form::Estimator::Params p;
    p.constraints = params_.constraints;
    p.extraction = params_.extraction;
    p.matcher = params_.matcher;
    p.map = params_.map;
    p.scans = params_.scans;
    p.num_threads = params_.num_threads;
    estimator_ = std::make_shared<form::Estimator>(p);
  }

  // Add an IMU measurement
  void add_imu(evalio::ImuMeasurement mm) override {
    if (fuse_imu_) {
      inertial_estimator_->register_imu(imu_to_form(mm));
    }
  }

  // Add a LiDAR measurement
  std::map<std::string, std::vector<evalio::Point>>
  add_lidar(evalio::LidarMeasurement mm) override {
    // convert to evalio
    std::vector<form::PointXYZf> scan;
    // send a stamp that is at the end of the scan
    auto end_ev = mm.stamp + delta_time_;
    auto end = form::Stamp{.sec = end_ev.sec, .nsec = end_ev.nsec};

    for (const auto &point : mm.points) {
      scan.push_back(point_to_form(point));
    }

    // run the estimator
    std::vector<form::PlanarFeat> planar_kp;
    std::vector<form::PointFeat> point_kp;
    if (fuse_imu_) {
      std::tie(planar_kp, point_kp) = inertial_estimator_->register_scan(scan, end);
      current_pose = pose_to_evalio(inertial_estimator_->current_estimate());
    } else {
      std::tie(planar_kp, point_kp) = estimator_->register_scan(scan);
      current_pose = pose_to_evalio(estimator_->current_lidar_estimate() *
                                    params_.imu.imu_T_lidar.inverse());
    }

    // extract the keypoints
    std::map<std::string, std::vector<evalio::Point>> points = {{"planar", {}},
                                                                {"point", {}}};
    auto &all_planar = points["planar"];
    auto &all_point = points["point"];
    for (const auto &point : planar_kp) {
      const auto ev_point = point_to_evalio(point);
      all_planar.push_back(ev_point);
    }

    for (const auto &point : point_kp) {
      const auto ev_point = point_to_evalio(point);
      all_point.push_back(ev_point);
    }

    return points;
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
                                evalio::LidarParams &lidar_params) {
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
