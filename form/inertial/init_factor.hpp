#pragma once

#include <gtsam/geometry/Unit3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

namespace form {

class GravityBiasPreintFactor
    : public gtsam::NoiseModelFactorN<gtsam::Vector3, gtsam::Unit3> {

private:
  // Preintegrated measurements
  gtsam::PreintegratedCombinedMeasurements pim_;

  // Initial and final poses and initial velocity
  gtsam::Pose3 x0_;
  gtsam::Pose3 x1_;
  gtsam::Velocity3 v0_;

public:
  GravityBiasPreintFactor(const gtsam::PreintegratedCombinedMeasurements &pim,
                          const gtsam::Pose3 &x0, const gtsam::Pose3 &x1,
                          const gtsam::Velocity3 &v0, const gtsam::Key &bias_key,
                          const gtsam::Key &gravity_key)
      : gtsam::NoiseModelFactorN<gtsam::Vector3, gtsam::Unit3>(
            gtsam::noiseModel::Unit::Create(3), bias_key, gravity_key),
        pim_(pim), x0_(x0), x1_(x1), v0_(v0) {}

  gtsam::Vector
  evaluateError(const gtsam::Vector3 &bias, const gtsam::Unit3 &gravity,
                boost::optional<gtsam::Matrix &> H1 = boost::none,
                boost::optional<gtsam::Matrix &> H2 = boost::none) const override;
};

} // namespace form