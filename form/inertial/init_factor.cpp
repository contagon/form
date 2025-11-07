#include "form/inertial/init_factor.hpp"
#include <gtsam/navigation/ImuBias.h>

namespace form {

gtsam::Vector GravityBiasPreintFactor::evaluateError(
    const gtsam::Vector3 &bias, const gtsam::Unit3 &gravity,
    boost::optional<gtsam::Matrix &> H1, boost::optional<gtsam::Matrix &> H2) const {

  const auto R0 = x0_.rotation();
  const auto p0 = x0_.translation();
  const auto p1 = x1_.translation();

  const gtsam::Vector3 gyro_b = pim_.biasHat().gyroscope();
  const auto b = gtsam::imuBias::ConstantBias(bias, gyro_b);
  const gtsam::Vector9 corr = pim_.biasCorrectedDelta(b);

  const double dt = pim_.deltaTij();
  const double grav_scale = 0.5 * dt * dt * 9.81;

  if (H1) {
    H1->resize(3, 3);
    *H1 = R0.matrix() * pim_.preintegrated_H_biasAcc().block<3, 3>(3, 0);
  }

  if (H2) {
    H2->resize(3, 2);
    *H2 = gravity.basis() * grav_scale;
  }

  const gtsam::Vector3 p1hat =
      R0.rotate(corr.segment(3, 3)) + p0 + v0_ * dt + grav_scale * gravity.point3();
  return p1hat - p1;
}
} // namespace form