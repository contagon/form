/*
To avoid mismatched versions of GTSAM and its Python bindings, to add
some custom bindings for some things we were experimenting with, and to switch
bindings to nanobind, we bind a small subset of gtsam that we need here.

These are largely slightly modified from the official gtsam pybind11 bindings
that are automatically generated when building gtsam python bindings.
*/

#include <cstdio>
#include <gtsam/base/utilities.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/nonlinear/Values.h>

#include <memory>
#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <utility>

namespace nb = nanobind;
namespace gt = gtsam;
using std::string;

// clang-format off
inline void make_gtsam_bindings(nb::module_ &m_) {
  nb::class_<gtsam::Rot3>(m_, "Rot3")
        .def(nb::init<>())
        .def(nb::init<const gtsam::Matrix&>(), nb::arg("R"))
        .def(nb::init<const gtsam::Point3&, const gtsam::Point3&, const gtsam::Point3&>(), nb::arg("col1"), nb::arg("col2"), nb::arg("col3"))
        .def(nb::init<double, double, double, double, double, double, double, double, double>(), nb::arg("R11"), nb::arg("R12"), nb::arg("R13"), nb::arg("R21"), nb::arg("R22"), nb::arg("R23"), nb::arg("R31"), nb::arg("R32"), nb::arg("R33"))
        .def(nb::init<double, double, double, double>(), nb::arg("w"), nb::arg("x"), nb::arg("y"), nb::arg("z"))
        .def("print",[](gtsam::Rot3* self, std::string s){ self->print(s);}, nb::arg("s") = "")
        .def("__repr__",
                    [](const gtsam::Rot3& self, std::string s){
                        gtsam::RedirectCout redirect;
                        self.print(s);
                        return redirect.str();
                    }, nb::arg("s") = "")
        .def("equals",[](gtsam::Rot3* self, const gtsam::Rot3& rot, double tol){return self->equals(rot, tol);}, nb::arg("rot"), nb::arg("tol"))
        .def("inverse",[](gtsam::Rot3* self){return self->inverse();})
        .def("compose",[](gtsam::Rot3* self, const gtsam::Rot3& p2){return self->compose(p2);}, nb::arg("p2"))
        .def("between",[](gtsam::Rot3* self, const gtsam::Rot3& p2){return self->between(p2);}, nb::arg("p2"))
        .def("retract",[](gtsam::Rot3* self, const gtsam::Vector& v){return self->retract(v);}, nb::arg("v"))
        .def("localCoordinates",[](gtsam::Rot3* self, const gtsam::Rot3& p){return self->localCoordinates(p);}, nb::arg("p"))
        .def("rotate",[](gtsam::Rot3* self, const gtsam::Point3& p){return self->rotate(p);}, nb::arg("p"))
        .def("unrotate",[](gtsam::Rot3* self, const gtsam::Point3& p){return self->unrotate(p);}, nb::arg("p"))
        .def("logmap",[](gtsam::Rot3* self, const gtsam::Rot3& p){return self->logmap(p);}, nb::arg("p"))
        .def("matrix",[](gtsam::Rot3* self){return self->matrix();})
        .def("transpose",[](gtsam::Rot3* self){return self->transpose();})
        .def("column",[](gtsam::Rot3* self, size_t index){return self->column(index);}, nb::arg("index"))
        .def("xyz",[](gtsam::Rot3* self){return self->xyz();})
        .def("ypr",[](gtsam::Rot3* self){return self->ypr();})
        .def("rpy",[](gtsam::Rot3* self){return self->rpy();})
        .def("roll",[](gtsam::Rot3* self){return self->roll();})
        .def("pitch",[](gtsam::Rot3* self){return self->pitch();})
        .def("yaw",[](gtsam::Rot3* self){return self->yaw();})
        // .def("axisAngle",[](gtsam::Rot3* self){return self->axisAngle();})
        // .def("toQuaternion",[](gtsam::Rot3* self){return self->toQuaternion();})
        .def("slerp",[](gtsam::Rot3* self, double t, const gtsam::Rot3& other){return self->slerp(t, other);}, nb::arg("t"), nb::arg("other"))
        .def_static("Rx",[](double t){return gtsam::Rot3::Rx(t);}, nb::arg("t"))
        .def_static("Ry",[](double t){return gtsam::Rot3::Ry(t);}, nb::arg("t"))
        .def_static("Rz",[](double t){return gtsam::Rot3::Rz(t);}, nb::arg("t"))
        .def_static("RzRyRx",[](double x, double y, double z){return gtsam::Rot3::RzRyRx(x, y, z);}, nb::arg("x"), nb::arg("y"), nb::arg("z"))
        .def_static("RzRyRx",[](const gtsam::Vector& xyz){return gtsam::Rot3::RzRyRx(xyz);}, nb::arg("xyz"))
        .def_static("Yaw",[](double t){return gtsam::Rot3::Yaw(t);}, nb::arg("t"))
        .def_static("Pitch",[](double t){return gtsam::Rot3::Pitch(t);}, nb::arg("t"))
        .def_static("Roll",[](double t){return gtsam::Rot3::Roll(t);}, nb::arg("t"))
        .def_static("Ypr",[](double y, double p, double r){return gtsam::Rot3::Ypr(y, p, r);}, nb::arg("y"), nb::arg("p"), nb::arg("r"))
        .def_static("Quaternion",[](double w, double x, double y, double z){return gtsam::Rot3::Quaternion(w, x, y, z);}, nb::arg("w"), nb::arg("x"), nb::arg("y"), nb::arg("z"))
        .def_static("AxisAngle",[](const gtsam::Point3& axis, double angle){return gtsam::Rot3::AxisAngle(axis, angle);}, nb::arg("axis"), nb::arg("angle"))
        .def_static("Rodrigues",[](const gtsam::Vector& v){return gtsam::Rot3::Rodrigues(v);}, nb::arg("v"))
        .def_static("Rodrigues",[](double wx, double wy, double wz){return gtsam::Rot3::Rodrigues(wx, wy, wz);}, nb::arg("wx"), nb::arg("wy"), nb::arg("wz"))
        .def_static("ClosestTo",[](const gtsam::Matrix& M){return gtsam::Rot3::ClosestTo(M);}, nb::arg("M"))
        .def_static("Identity",[](){return gtsam::Rot3::Identity();})
        .def_static("Expmap",[](const gtsam::Vector& v){return gtsam::Rot3::Expmap(v);}, nb::arg("v"))
        .def_static("Logmap",[](const gtsam::Rot3& p){return gtsam::Rot3::Logmap(p);}, nb::arg("p"))
        .def(nb::self * nb::self);


    nb::class_<gtsam::Pose3>(m_, "Pose3")
        .def(nb::init<>())
        .def(nb::init<const gtsam::Pose3&>(), nb::arg("other"))
        .def(nb::init<const gtsam::Rot3&, const gtsam::Point3&>(), nb::arg("r"), nb::arg("t"))
        .def(nb::init<const gtsam::Matrix&>(), nb::arg("mat"))
        .def("print",[](gtsam::Pose3* self, std::string s){ self->print(s);}, nb::arg("s") = "")
        .def("__repr__",
                    [](const gtsam::Pose3& self, std::string s){
                        gtsam::RedirectCout redirect;
                        self.print(s);
                        return redirect.str();
                    }, nb::arg("s") = "")
        .def("equals",[](gtsam::Pose3* self, const gtsam::Pose3& pose, double tol){return self->equals(pose, tol);}, nb::arg("pose"), nb::arg("tol"))
        .def("inverse",[](gtsam::Pose3* self){return self->inverse();})
        .def("inverse",[](gtsam::Pose3* self, Eigen::Ref<Eigen::MatrixXd> H){return self->inverse(H);}, nb::arg("H"))
        .def("compose",[](gtsam::Pose3* self, const gtsam::Pose3& pose){return self->compose(pose);}, nb::arg("pose"))
        .def("compose",[](gtsam::Pose3* self, const gtsam::Pose3& pose, Eigen::Ref<Eigen::MatrixXd> H1, Eigen::Ref<Eigen::MatrixXd> H2){return self->compose(pose, H1, H2);}, nb::arg("pose"), nb::arg("H1"), nb::arg("H2"))
        .def("between",[](gtsam::Pose3* self, const gtsam::Pose3& pose){return self->between(pose);}, nb::arg("pose"))
        .def("between",[](gtsam::Pose3* self, const gtsam::Pose3& pose, Eigen::Ref<Eigen::MatrixXd> H1, Eigen::Ref<Eigen::MatrixXd> H2){return self->between(pose, H1, H2);}, nb::arg("pose"), nb::arg("H1"), nb::arg("H2"))
        .def("slerp",[](gtsam::Pose3* self, double t, const gtsam::Pose3& pose){return self->slerp(t, pose);}, nb::arg("t"), nb::arg("pose"))
        .def("slerp",[](gtsam::Pose3* self, double t, const gtsam::Pose3& pose, Eigen::Ref<Eigen::MatrixXd> Hx, Eigen::Ref<Eigen::MatrixXd> Hy){return self->slerp(t, pose, Hx, Hy);}, nb::arg("t"), nb::arg("pose"), nb::arg("Hx"), nb::arg("Hy"))
        .def("retract",[](gtsam::Pose3* self, const gtsam::Vector& v){return self->retract(v);}, nb::arg("v"))
        .def("retract",[](gtsam::Pose3* self, const gtsam::Vector& v, Eigen::Ref<Eigen::MatrixXd> Hxi){return self->retract(v, Hxi);}, nb::arg("v"), nb::arg("Hxi"))
        .def("localCoordinates",[](gtsam::Pose3* self, const gtsam::Pose3& pose){return self->localCoordinates(pose);}, nb::arg("pose"))
        .def("localCoordinates",[](gtsam::Pose3* self, const gtsam::Pose3& pose, Eigen::Ref<Eigen::MatrixXd> Hxi){return self->localCoordinates(pose, Hxi);}, nb::arg("pose"), nb::arg("Hxi"))
        .def("expmap",[](gtsam::Pose3* self, const gtsam::Vector& v){return self->expmap(v);}, nb::arg("v"))
        .def("logmap",[](gtsam::Pose3* self, const gtsam::Pose3& pose){return self->logmap(pose);}, nb::arg("pose"))
        .def("AdjointMap",[](gtsam::Pose3* self){return self->AdjointMap();})
        .def("Adjoint",[](gtsam::Pose3* self, const gtsam::Vector& xi){return self->Adjoint(xi);}, nb::arg("xi"))
        .def("Adjoint",[](gtsam::Pose3* self, const gtsam::Vector& xi, Eigen::Ref<Eigen::MatrixXd> H_this, Eigen::Ref<Eigen::MatrixXd> H_xib){return self->Adjoint(xi, H_this, H_xib);}, nb::arg("xi"), nb::arg("H_this"), nb::arg("H_xib"))
        .def("AdjointTranspose",[](gtsam::Pose3* self, const gtsam::Vector& xi){return self->AdjointTranspose(xi);}, nb::arg("xi"))
        .def("AdjointTranspose",[](gtsam::Pose3* self, const gtsam::Vector& xi, Eigen::Ref<Eigen::MatrixXd> H_this, Eigen::Ref<Eigen::MatrixXd> H_x){return self->AdjointTranspose(xi, H_this, H_x);}, nb::arg("xi"), nb::arg("H_this"), nb::arg("H_x"))
        .def("transformFrom",[](gtsam::Pose3* self, const gtsam::Point3& point){return self->transformFrom(point);}, nb::arg("point"))
        .def("transformFrom",[](gtsam::Pose3* self, const gtsam::Point3& point, Eigen::Ref<Eigen::MatrixXd> Hself, Eigen::Ref<Eigen::MatrixXd> Hpoint){return self->transformFrom(point, Hself, Hpoint);}, nb::arg("point"), nb::arg("Hself"), nb::arg("Hpoint"))
        .def("transformTo",[](gtsam::Pose3* self, const gtsam::Point3& point){return self->transformTo(point);}, nb::arg("point"))
        .def("transformTo",[](gtsam::Pose3* self, const gtsam::Point3& point, Eigen::Ref<Eigen::MatrixXd> Hself, Eigen::Ref<Eigen::MatrixXd> Hpoint){return self->transformTo(point, Hself, Hpoint);}, nb::arg("point"), nb::arg("Hself"), nb::arg("Hpoint"))
        .def("transformFrom",[](gtsam::Pose3* self, const gtsam::Matrix& points){return self->transformFrom(points);}, nb::arg("points"))
        .def("transformTo",[](gtsam::Pose3* self, const gtsam::Matrix& points){return self->transformTo(points);}, nb::arg("points"))
        .def("rotation",[](gtsam::Pose3* self){return self->rotation();})
        .def("rotation",[](gtsam::Pose3* self, Eigen::Ref<Eigen::MatrixXd> Hself){return self->rotation(Hself);}, nb::arg("Hself"))
        .def("translation",[](gtsam::Pose3* self){return self->translation();})
        .def("translation",[](gtsam::Pose3* self, Eigen::Ref<Eigen::MatrixXd> Hself){return self->translation(Hself);}, nb::arg("Hself"))
        .def("x",[](gtsam::Pose3* self){return self->x();})
        .def("y",[](gtsam::Pose3* self){return self->y();})
        .def("z",[](gtsam::Pose3* self){return self->z();})
        .def("matrix",[](gtsam::Pose3* self){return self->matrix();})
        .def("transformPoseFrom",[](gtsam::Pose3* self, const gtsam::Pose3& pose){return self->transformPoseFrom(pose);}, nb::arg("pose"))
        .def("transformPoseFrom",[](gtsam::Pose3* self, const gtsam::Pose3& pose, Eigen::Ref<Eigen::MatrixXd> Hself, Eigen::Ref<Eigen::MatrixXd> HaTb){return self->transformPoseFrom(pose, Hself, HaTb);}, nb::arg("pose"), nb::arg("Hself"), nb::arg("HaTb"))
        .def("transformPoseTo",[](gtsam::Pose3* self, const gtsam::Pose3& pose){return self->transformPoseTo(pose);}, nb::arg("pose"))
        .def("transformPoseTo",[](gtsam::Pose3* self, const gtsam::Pose3& pose, Eigen::Ref<Eigen::MatrixXd> Hself, Eigen::Ref<Eigen::MatrixXd> HwTb){return self->transformPoseTo(pose, Hself, HwTb);}, nb::arg("pose"), nb::arg("Hself"), nb::arg("HwTb"))
        .def("range",[](gtsam::Pose3* self, const gtsam::Point3& point){return self->range(point);}, nb::arg("point"))
        .def("range",[](gtsam::Pose3* self, const gtsam::Point3& point, Eigen::Ref<Eigen::MatrixXd> Hself, Eigen::Ref<Eigen::MatrixXd> Hpoint){return self->range(point, Hself, Hpoint);}, nb::arg("point"), nb::arg("Hself"), nb::arg("Hpoint"))
        .def("range",[](gtsam::Pose3* self, const gtsam::Pose3& pose){return self->range(pose);}, nb::arg("pose"))
        .def("range",[](gtsam::Pose3* self, const gtsam::Pose3& pose, Eigen::Ref<Eigen::MatrixXd> Hself, Eigen::Ref<Eigen::MatrixXd> Hpose){return self->range(pose, Hself, Hpose);}, nb::arg("pose"), nb::arg("Hself"), nb::arg("Hpose"))
        .def_static("Identity",[](){return gtsam::Pose3::Identity();})
        .def_static("Expmap",[](const gtsam::Vector& v){return gtsam::Pose3::Expmap(v);}, nb::arg("v"))
        .def_static("Expmap",[](const gtsam::Vector& v, Eigen::Ref<Eigen::MatrixXd> Hxi){return gtsam::Pose3::Expmap(v, Hxi);}, nb::arg("v"), nb::arg("Hxi"))
        .def_static("Logmap",[](const gtsam::Pose3& pose){return gtsam::Pose3::Logmap(pose);}, nb::arg("pose"))
        .def_static("Logmap",[](const gtsam::Pose3& pose, Eigen::Ref<Eigen::MatrixXd> Hpose){return gtsam::Pose3::Logmap(pose, Hpose);}, nb::arg("pose"), nb::arg("Hpose"))
        .def_static("adjointMap",[](const gtsam::Vector& xi){return gtsam::Pose3::adjointMap(xi);}, nb::arg("xi"))
        .def_static("adjoint",[](const gtsam::Vector& xi, const gtsam::Vector& y){return gtsam::Pose3::adjoint(xi, y);}, nb::arg("xi"), nb::arg("y"))
        .def_static("adjointMap_",[](const gtsam::Vector& xi){return gtsam::Pose3::adjointMap_(xi);}, nb::arg("xi"))
        .def_static("adjoint_",[](const gtsam::Vector& xi, const gtsam::Vector& y){return gtsam::Pose3::adjoint_(xi, y);}, nb::arg("xi"), nb::arg("y"))
        .def_static("adjointTranspose",[](const gtsam::Vector& xi, const gtsam::Vector& y){return gtsam::Pose3::adjointTranspose(xi, y);}, nb::arg("xi"), nb::arg("y"))
        .def_static("ExpmapDerivative",[](const gtsam::Vector& xi){return gtsam::Pose3::ExpmapDerivative(xi);}, nb::arg("xi"))
        .def_static("LogmapDerivative",[](const gtsam::Pose3& xi){return gtsam::Pose3::LogmapDerivative(xi);}, nb::arg("xi"))
        .def_static("wedge",[](double wx, double wy, double wz, double vx, double vy, double vz){return gtsam::Pose3::wedge(wx, wy, wz, vx, vy, vz);}, nb::arg("wx"), nb::arg("wy"), nb::arg("wz"), nb::arg("vx"), nb::arg("vy"), nb::arg("vz"))
        .def(nb::self * nb::self);

    nb::class_<gtsam::imuBias::ConstantBias>(m_, "ConstantBias")
        .def(nb::init<>())
        .def(nb::init<const gtsam::Vector&, const gtsam::Vector&>(), nb::arg("biasAcc"), nb::arg("biasGyro"))
        .def("print",[](gtsam::imuBias::ConstantBias* self, string s){ ; self->print(s);}, nb::arg("s") = "")
        .def("__repr__",
                    [](const gtsam::imuBias::ConstantBias& self, string s){
                        gtsam::RedirectCout redirect;
                        self.print(s);
                        return redirect.str();
                    }, nb::arg("s") = "")
        .def("equals",[](gtsam::imuBias::ConstantBias* self, const gtsam::imuBias::ConstantBias& expected, double tol){return self->equals(expected, tol);}, nb::arg("expected"), nb::arg("tol"))
        .def("vector",[](gtsam::imuBias::ConstantBias* self){return self->vector();})
        .def("accelerometer",[](gtsam::imuBias::ConstantBias* self){return self->accelerometer();})
        .def("gyroscope",[](gtsam::imuBias::ConstantBias* self){return self->gyroscope();})
        .def("correctAccelerometer",[](gtsam::imuBias::ConstantBias* self, const gtsam::Vector& measurement){return self->correctAccelerometer(measurement);}, nb::arg("measurement"))
        .def("correctGyroscope",[](gtsam::imuBias::ConstantBias* self, const gtsam::Vector& measurement){return self->correctGyroscope(measurement);}, nb::arg("measurement"))
        .def_static("Identity",[](){return gtsam::imuBias::ConstantBias::Identity();})
        .def(-nb::self)
        .def(nb::self + nb::self)
        .def(nb::self - nb::self);

    nb::class_<gtsam::NavState>(m_, "NavState")
        .def(nb::init<>())
        .def(nb::init<const gtsam::Rot3&, const gtsam::Point3&, const gtsam::Vector&>(), nb::arg("R"), nb::arg("t"), nb::arg("v"))
        .def(nb::init<const gtsam::Pose3&, const gtsam::Vector&>(), nb::arg("pose"), nb::arg("v"))
        .def("print",[](gtsam::NavState* self, string s){ self->print(s);}, nb::arg("s") = "")
        .def("__repr__",
                    [](const gtsam::NavState& self, string s){
                        gtsam::RedirectCout redirect;
                        self.print(s);
                        return redirect.str();
                    }, nb::arg("s") = "")
        .def("equals",[](gtsam::NavState* self, const gtsam::NavState& expected, double tol){return self->equals(expected, tol);}, nb::arg("expected"), nb::arg("tol"))
        .def("attitude",[](gtsam::NavState* self){return self->attitude();})
        .def("position",[](gtsam::NavState* self){return self->position();})
        .def("velocity",[](gtsam::NavState* self){return self->velocity();})
        .def("pose",[](gtsam::NavState* self){return self->pose();})
        .def("retract",[](gtsam::NavState* self, const gtsam::Vector& x){return self->retract(x);}, nb::arg("x"))
        .def("correctPIM", [](gtsam::NavState* self, const gtsam::Vector9& pimCorrection, double dt, const gtsam::Vector3& gravity){return self->correctPIM(pimCorrection, dt, gravity, boost::none);})
        .def("localCoordinates",[](gtsam::NavState* self, const gtsam::NavState& g){return self->localCoordinates(g);}, nb::arg("g"));

    nb::class_<gtsam::Values>(m_, "Values")
        .def(nb::init<>())
        .def(nb::init<const gtsam::Values&>(), nb::arg("other"))
        .def("size",[](gtsam::Values* self){return self->size();})
        .def("empty",[](gtsam::Values* self){return self->empty();})
        .def("clear",[](gtsam::Values* self){ self->clear();})
        .def("dim",[](gtsam::Values* self){return self->dim();})
        .def("print",[](gtsam::Values* self, string s){ self->print(s, gtsam::DefaultKeyFormatter);}, nb::arg("s") = "")
        .def("__repr__",
                    [](const gtsam::Values& self, string s){
                        gtsam::RedirectCout redirect;
                        self.print(s, gtsam::DefaultKeyFormatter);
                        return redirect.str();
                    }, nb::arg("s") = "")
        .def("equals",[](gtsam::Values* self, const gtsam::Values& other, double tol){return self->equals(other, tol);}, nb::arg("other"), nb::arg("tol"))
        .def("insert",[](gtsam::Values* self, const gtsam::Values& values){ self->insert(values);}, nb::arg("values"))
        .def("update",[](gtsam::Values* self, const gtsam::Values& values){ self->update(values);}, nb::arg("values"))
        .def("insert_or_assign",[](gtsam::Values* self, const gtsam::Values& values){ self->insert_or_assign(values);}, nb::arg("values"))
        .def("erase",[](gtsam::Values* self, size_t j){ self->erase(j);}, nb::arg("j"))
        .def("swap",[](gtsam::Values* self, gtsam::Values& values){ self->swap(values);}, nb::arg("values"))
        .def("exists",[](gtsam::Values* self, size_t j){return self->exists(j);}, nb::arg("j"))
        .def("keys",[](gtsam::Values* self){return self->keys();})
        // .def("retract",[](gtsam::Values* self, const gtsam::VectorValues& delta){return self->retract(delta);}, nb::arg("delta"))
        .def("insert_vector",[](gtsam::Values* self, size_t j, const gtsam::Vector& vector){ self->insert(j, vector);}, nb::arg("j"), nb::arg("vector"))
        .def("insert",[](gtsam::Values* self, size_t j, const gtsam::Vector& vector){ self->insert(j, vector);}, nb::arg("j"), nb::arg("vector"))
        .def("insert_matrix",[](gtsam::Values* self, size_t j, const gtsam::Matrix& matrix){ self->insert(j, matrix);}, nb::arg("j"), nb::arg("matrix"))
        .def("insert",[](gtsam::Values* self, size_t j, const gtsam::Matrix& matrix){ self->insert(j, matrix);}, nb::arg("j"), nb::arg("matrix"))
        .def("insert_point2",[](gtsam::Values* self, size_t j, const gtsam::Point2& point2){ self->insert(j, point2);}, nb::arg("j"), nb::arg("point2"))
        .def("insert",[](gtsam::Values* self, size_t j, const gtsam::Point2& point2){ self->insert(j, point2);}, nb::arg("j"), nb::arg("point2"))
        .def("insert_point3",[](gtsam::Values* self, size_t j, const gtsam::Point3& point3){ self->insert(j, point3);}, nb::arg("j"), nb::arg("point3"))
        .def("insert",[](gtsam::Values* self, size_t j, const gtsam::Point3& point3){ self->insert(j, point3);}, nb::arg("j"), nb::arg("point3"))
        .def("insert_rot3",[](gtsam::Values* self, size_t j, const gtsam::Rot3& rot3){ self->insert(j, rot3);}, nb::arg("j"), nb::arg("rot3"))
        .def("insert",[](gtsam::Values* self, size_t j, const gtsam::Rot3& rot3){ self->insert(j, rot3);}, nb::arg("j"), nb::arg("rot3"))
        .def("insert_pose3",[](gtsam::Values* self, size_t j, const gtsam::Pose3& pose3){ self->insert(j, pose3);}, nb::arg("j"), nb::arg("pose3"))
        .def("insert",[](gtsam::Values* self, size_t j, const gtsam::Pose3& pose3){ self->insert(j, pose3);}, nb::arg("j"), nb::arg("pose3"))
        .def("insert_constant_bias",[](gtsam::Values* self, size_t j, const gtsam::imuBias::ConstantBias& constant_bias){ self->insert(j, constant_bias);}, nb::arg("j"), nb::arg("constant_bias"))
        .def("insert",[](gtsam::Values* self, size_t j, const gtsam::imuBias::ConstantBias& constant_bias){ self->insert(j, constant_bias);}, nb::arg("j"), nb::arg("constant_bias"))
        .def("insert_nav_state",[](gtsam::Values* self, size_t j, const gtsam::NavState& nav_state){ self->insert(j, nav_state);}, nb::arg("j"), nb::arg("nav_state"))
        .def("insert",[](gtsam::Values* self, size_t j, const gtsam::NavState& nav_state){ self->insert(j, nav_state);}, nb::arg("j"), nb::arg("nav_state"))
        .def("insert_c",[](gtsam::Values* self, size_t j, double c){ self->insert(j, c);}, nb::arg("j"), nb::arg("c"))
        .def("insert",[](gtsam::Values* self, size_t j, double c){ self->insert(j, c);}, nb::arg("j"), nb::arg("c"))
        .def("insertPoint2",[](gtsam::Values* self, size_t j, const gtsam::Point2& val){ self->insert<gtsam::Point2>(j, val);}, nb::arg("j"), nb::arg("val"))
        .def("insertPoint3",[](gtsam::Values* self, size_t j, const gtsam::Point3& val){ self->insert<gtsam::Point3>(j, val);}, nb::arg("j"), nb::arg("val"))
        .def("update",[](gtsam::Values* self, size_t j, const gtsam::Point2& point2){ self->update(j, point2);}, nb::arg("j"), nb::arg("point2"))
        .def("update",[](gtsam::Values* self, size_t j, const gtsam::Point3& point3){ self->update(j, point3);}, nb::arg("j"), nb::arg("point3"))
        .def("update",[](gtsam::Values* self, size_t j, const gtsam::Rot3& rot3){ self->update(j, rot3);}, nb::arg("j"), nb::arg("rot3"))
        .def("update",[](gtsam::Values* self, size_t j, const gtsam::Pose3& pose3){ self->update(j, pose3);}, nb::arg("j"), nb::arg("pose3"))
        .def("update",[](gtsam::Values* self, size_t j, const gtsam::imuBias::ConstantBias& constant_bias){ self->update(j, constant_bias);}, nb::arg("j"), nb::arg("constant_bias"))
        .def("update",[](gtsam::Values* self, size_t j, const gtsam::NavState& nav_state){ self->update(j, nav_state);}, nb::arg("j"), nb::arg("nav_state"))
        .def("update",[](gtsam::Values* self, size_t j, const gtsam::Vector& vector){ self->update(j, vector);}, nb::arg("j"), nb::arg("vector"))
        .def("update",[](gtsam::Values* self, size_t j, const gtsam::Matrix& matrix){ self->update(j, matrix);}, nb::arg("j"), nb::arg("matrix"))
        .def("update",[](gtsam::Values* self, size_t j, double c){ self->update(j, c);}, nb::arg("j"), nb::arg("c"))
        .def("insert_or_assign",[](gtsam::Values* self, size_t j, const gtsam::Point2& point2){ self->insert_or_assign(j, point2);}, nb::arg("j"), nb::arg("point2"))
        .def("insert_or_assign",[](gtsam::Values* self, size_t j, const gtsam::Point3& point3){ self->insert_or_assign(j, point3);}, nb::arg("j"), nb::arg("point3"))
        .def("insert_or_assign",[](gtsam::Values* self, size_t j, const gtsam::Rot3& rot3){ self->insert_or_assign(j, rot3);}, nb::arg("j"), nb::arg("rot3"))
        .def("insert_or_assign",[](gtsam::Values* self, size_t j, const gtsam::Pose3& pose3){ self->insert_or_assign(j, pose3);}, nb::arg("j"), nb::arg("pose3"))
        .def("insert_or_assign",[](gtsam::Values* self, size_t j, const gtsam::imuBias::ConstantBias& constant_bias){ self->insert_or_assign(j, constant_bias);}, nb::arg("j"), nb::arg("constant_bias"))
        .def("insert_or_assign",[](gtsam::Values* self, size_t j, const gtsam::NavState& nav_state){ self->insert_or_assign(j, nav_state);}, nb::arg("j"), nb::arg("nav_state"))
        .def("insert_or_assign",[](gtsam::Values* self, size_t j, const gtsam::Vector& vector){ self->insert_or_assign(j, vector);}, nb::arg("j"), nb::arg("vector"))
        .def("insert_or_assign",[](gtsam::Values* self, size_t j, const gtsam::Matrix& matrix){ self->insert_or_assign(j, matrix);}, nb::arg("j"), nb::arg("matrix"))
        .def("insert_or_assign",[](gtsam::Values* self, size_t j, double c){ self->insert_or_assign(j, c);}, nb::arg("j"), nb::arg("c"))
        .def("atPoint2",[](gtsam::Values* self, size_t j){return self->at<gtsam::Point2>(j);}, nb::arg("j"))
        .def("atPoint3",[](gtsam::Values* self, size_t j){return self->at<gtsam::Point3>(j);}, nb::arg("j"))
        .def("atRot3",[](gtsam::Values* self, size_t j){return self->at<gtsam::Rot3>(j);}, nb::arg("j"))
        .def("atPose3",[](gtsam::Values* self, size_t j){return self->at<gtsam::Pose3>(j);}, nb::arg("j"))
        .def("atConstantBias",[](gtsam::Values* self, size_t j){return self->at<gtsam::imuBias::ConstantBias>(j);}, nb::arg("j"))
        .def("atNavState",[](gtsam::Values* self, size_t j){return self->at<gtsam::NavState>(j);}, nb::arg("j"))
        .def("atVector",[](gtsam::Values* self, size_t j){return self->at<gtsam::Vector>(j);}, nb::arg("j"))
        .def("atMatrix",[](gtsam::Values* self, size_t j){return self->at<gtsam::Matrix>(j);}, nb::arg("j"))
        .def("atDouble",[](gtsam::Values* self, size_t j){return self->at<double>(j);}, nb::arg("j"));

      nb::class_<gtsam::Symbol>(m_, "Symbol")
        .def(nb::init<>())
        .def(nb::init<unsigned char, uint64_t>(), nb::arg("c"), nb::arg("j"))
        .def(nb::init<size_t>(), nb::arg("key"))
        .def("key",[](gtsam::Symbol* self){return self->key();})
        .def("print",[](gtsam::Symbol* self, const string& s){ self->print(s);}, nb::arg("s") = "")
        .def("__repr__",
                    [](const gtsam::Symbol& self, const string& s){
                        gtsam::RedirectCout redirect;
                        self.print(s);
                        return redirect.str();
                    }, nb::arg("s") = "")
        .def("equals",[](gtsam::Symbol* self, const gtsam::Symbol& expected, double tol){return self->equals(expected, tol);}, nb::arg("expected"), nb::arg("tol"))
        .def("chr",[](gtsam::Symbol* self){return self->chr();})
        .def("index",[](gtsam::Symbol* self){return self->index();})
        .def("string",[](gtsam::Symbol* self){return self->string();});

    m_.def("A",[](size_t j){return gtsam::symbol_shorthand::A(j);}, nb::arg("j"));
    m_.def("B",[](size_t j){return gtsam::symbol_shorthand::B(j);}, nb::arg("j"));
    m_.def("C",[](size_t j){return gtsam::symbol_shorthand::C(j);}, nb::arg("j"));
    m_.def("D",[](size_t j){return gtsam::symbol_shorthand::D(j);}, nb::arg("j"));
    m_.def("E",[](size_t j){return gtsam::symbol_shorthand::E(j);}, nb::arg("j"));
    m_.def("F",[](size_t j){return gtsam::symbol_shorthand::F(j);}, nb::arg("j"));
    m_.def("G",[](size_t j){return gtsam::symbol_shorthand::G(j);}, nb::arg("j"));
    m_.def("H",[](size_t j){return gtsam::symbol_shorthand::H(j);}, nb::arg("j"));
    m_.def("I",[](size_t j){return gtsam::symbol_shorthand::I(j);}, nb::arg("j"));
    m_.def("J",[](size_t j){return gtsam::symbol_shorthand::J(j);}, nb::arg("j"));
    m_.def("K",[](size_t j){return gtsam::symbol_shorthand::K(j);}, nb::arg("j"));
    m_.def("L",[](size_t j){return gtsam::symbol_shorthand::L(j);}, nb::arg("j"));
    m_.def("M",[](size_t j){return gtsam::symbol_shorthand::M(j);}, nb::arg("j"));
    m_.def("N",[](size_t j){return gtsam::symbol_shorthand::N(j);}, nb::arg("j"));
    m_.def("O",[](size_t j){return gtsam::symbol_shorthand::O(j);}, nb::arg("j"));
    m_.def("P",[](size_t j){return gtsam::symbol_shorthand::P(j);}, nb::arg("j"));
    m_.def("Q",[](size_t j){return gtsam::symbol_shorthand::Q(j);}, nb::arg("j"));
    m_.def("R",[](size_t j){return gtsam::symbol_shorthand::R(j);}, nb::arg("j"));
    m_.def("S",[](size_t j){return gtsam::symbol_shorthand::S(j);}, nb::arg("j"));
    m_.def("T",[](size_t j){return gtsam::symbol_shorthand::T(j);}, nb::arg("j"));
    m_.def("U",[](size_t j){return gtsam::symbol_shorthand::U(j);}, nb::arg("j"));
    m_.def("V",[](size_t j){return gtsam::symbol_shorthand::V(j);}, nb::arg("j"));
    m_.def("W",[](size_t j){return gtsam::symbol_shorthand::W(j);}, nb::arg("j"));
    m_.def("X",[](size_t j){return gtsam::symbol_shorthand::X(j);}, nb::arg("j"));
    m_.def("Y",[](size_t j){return gtsam::symbol_shorthand::Y(j);}, nb::arg("j"));
    m_.def("Z",[](size_t j){return gtsam::symbol_shorthand::Z(j);}, nb::arg("j"));


    nb::class_<gtsam::PreintegratedRotationParams>(m_, "PreintegratedRotationParams")
        .def(nb::init<>())
        .def("print",[](gtsam::PreintegratedRotationParams* self, string s){ self->print(s);}, nb::arg("s") = "")
        .def("__repr__",
                    [](const gtsam::PreintegratedRotationParams& self, string s){
                        gtsam::RedirectCout redirect;
                        self.print(s);
                        return redirect.str();
                    }, nb::arg("s") = "")
        // .def("equals",[](gtsam::PreintegratedRotationParams* self, const gtsam::PreintegratedRotationParams& expected, double tol){return self->equals(expected, tol);}, nb::arg("expected"), nb::arg("tol"))
        .def("setGyroscopeCovariance",[](gtsam::PreintegratedRotationParams* self, const gtsam::Matrix& cov){ self->setGyroscopeCovariance(cov);}, nb::arg("cov"))
        .def("setOmegaCoriolis",[](gtsam::PreintegratedRotationParams* self, const gtsam::Vector& omega){ self->setOmegaCoriolis(omega);}, nb::arg("omega"))
        // .def("setBodyPSensor",[](gtsam::PreintegratedRotationParams* self, const gtsam::Pose3& pose){ self->setBodyPSensor(pose);}, nb::arg("pose"))
        .def("getGyroscopeCovariance",[](gtsam::PreintegratedRotationParams* self){return self->getGyroscopeCovariance();})
        // .def("getOmegaCoriolis",[](gtsam::PreintegratedRotationParams* self){return self->getOmegaCoriolis();})
        // .def("getBodyPSensor",[](gtsam::PreintegratedRotationParams* self){return self->getBodyPSensor();})
        ;

    nb::class_<gtsam::PreintegrationParams, gtsam::PreintegratedRotationParams>(m_, "PreintegrationParams")
        .def(nb::init<const gtsam::Vector&>(), nb::arg("n_gravity"))
        .def("print",[](gtsam::PreintegrationParams* self, string s){ self->print(s);}, nb::arg("s") = "")
        .def("__repr__",
                    [](const gtsam::PreintegrationParams& self, string s){
                        gtsam::RedirectCout redirect;
                        self.print(s);
                        return redirect.str();
                    }, nb::arg("s") = "")
        .def("equals",[](gtsam::PreintegrationParams* self, const gtsam::PreintegrationParams& expected, double tol){return self->equals(expected, tol);}, nb::arg("expected"), nb::arg("tol"))
        .def("setAccelerometerCovariance",[](gtsam::PreintegrationParams* self, const gtsam::Matrix& cov){ self->setAccelerometerCovariance(cov);}, nb::arg("cov"))
        .def("setIntegrationCovariance",[](gtsam::PreintegrationParams* self, const gtsam::Matrix& cov){ self->setIntegrationCovariance(cov);}, nb::arg("cov"))
        .def("setUse2ndOrderCoriolis",[](gtsam::PreintegrationParams* self, bool flag){ self->setUse2ndOrderCoriolis(flag);}, nb::arg("flag"))
        .def("getAccelerometerCovariance",[](gtsam::PreintegrationParams* self){return self->getAccelerometerCovariance();})
        .def("getIntegrationCovariance",[](gtsam::PreintegrationParams* self){return self->getIntegrationCovariance();})
        .def("getUse2ndOrderCoriolis",[](gtsam::PreintegrationParams* self){return self->getUse2ndOrderCoriolis();})
        .def("serialize", [](gtsam::PreintegrationParams* self){ return gtsam::serialize(*self); })
        .def("deserialize", [](gtsam::PreintegrationParams* self, string serialized){ gtsam::deserialize(serialized, *self); }, nb::arg("serialized"))
        // .def_static("MakeSharedD",[](double g){return gtsam::PreintegrationParams::MakeSharedD(g);}, nb::arg("g"))
        // .def_static("MakeSharedU",[](double g){return gtsam::PreintegrationParams::MakeSharedU(g);}, nb::arg("g"))
        // .def_static("MakeSharedD",[](){return gtsam::PreintegrationParams::MakeSharedD();})
        // .def_static("MakeSharedU",[](){return gtsam::PreintegrationParams::MakeSharedU();})
        .def_rw("n_gravity", &gtsam::PreintegrationParams::n_gravity);

    nb::class_<gtsam::PreintegrationCombinedParams, gtsam::PreintegrationParams>(m_, "PreintegrationCombinedParams")
        .def(nb::init<const gtsam::Vector&>(), nb::arg("n_gravity"))
        .def("print",[](gtsam::PreintegrationCombinedParams* self, string s){ self->print(s);}, nb::arg("s") = "")
        .def("__repr__",
                    [](const gtsam::PreintegrationCombinedParams& self, string s){
                        gtsam::RedirectCout redirect;
                        self.print(s);
                        return redirect.str();
                    }, nb::arg("s") = "")
        // .def("equals",[](gtsam::PreintegrationCombinedParams* self, const gtsam::PreintegrationCombinedParams& expected, double tol){return self->equals(expected, tol);}, nb::arg("expected"), nb::arg("tol"))
        .def("setBiasAccCovariance",[](gtsam::PreintegrationCombinedParams* self, const gtsam::Matrix& cov){ self->setBiasAccCovariance(cov);}, nb::arg("cov"))
        .def("setBiasOmegaCovariance",[](gtsam::PreintegrationCombinedParams* self, const gtsam::Matrix& cov){ self->setBiasOmegaCovariance(cov);}, nb::arg("cov"))
        .def("setBiasAccOmegaInit",[](gtsam::PreintegrationCombinedParams* self, const gtsam::Matrix& cov){ self->setBiasAccOmegaInit(cov);}, nb::arg("cov"))
        .def("getBiasAccCovariance",[](gtsam::PreintegrationCombinedParams* self){return self->getBiasAccCovariance();})
        .def("getBiasOmegaCovariance",[](gtsam::PreintegrationCombinedParams* self){return self->getBiasOmegaCovariance();})
        .def("getBiasAccOmegaInit",[](gtsam::PreintegrationCombinedParams* self){return self->getBiasAccOmegaInit();})
        // .def_static("MakeSharedD",[](double g){return gtsam::PreintegrationCombinedParams::MakeSharedD(g);}, nb::arg("g"))
        // .def_static("MakeSharedU",[](double g){return gtsam::PreintegrationCombinedParams::MakeSharedU(g);}, nb::arg("g"))
        // .def_static("MakeSharedD",[](){return gtsam::PreintegrationCombinedParams::MakeSharedD();})
        // .def_static("MakeSharedU",[](){return gtsam::PreintegrationCombinedParams::MakeSharedU();})
        ;

    nb::class_<gtsam::PreintegratedCombinedMeasurements>(m_, "PreintegratedCombinedMeasurements")
        .def("__init__", [](gtsam::PreintegratedCombinedMeasurements* self, const gtsam::PreintegrationCombinedParams& params) {
            new (self) gtsam::PreintegratedCombinedMeasurements(boost::make_shared<gtsam::PreintegrationCombinedParams>(params));
        }, nb::arg("params"))
        .def("__init__", [](gtsam::PreintegratedCombinedMeasurements* self, const gtsam::PreintegrationCombinedParams& params, const gtsam::imuBias::ConstantBias& bias) {
            new (self) gtsam::PreintegratedCombinedMeasurements(boost::make_shared<gtsam::PreintegrationCombinedParams>(params), bias);
        }, nb::arg("params"), nb::arg("bias"))
        // .def(nb::init<const boost::shared_ptr<gtsam::PreintegrationCombinedParams>>(), nb::arg("params"))
        // .def(nb::init<const boost::shared_ptr<gtsam::PreintegrationCombinedParams>, const gtsam::imuBias::ConstantBias&>(), nb::arg("params"), nb::arg("bias"))
        .def("print",[](gtsam::PreintegratedCombinedMeasurements* self, string s){ self->print(s);}, nb::arg("s") = "Preintegrated Measurements:")
        .def("__repr__",
                    [](const gtsam::PreintegratedCombinedMeasurements& self, string s){
                        gtsam::RedirectCout redirect;
                        self.print(s);
                        return redirect.str();
                    }, nb::arg("s") = "Preintegrated Measurements:")
        // .def("equals",[](gtsam::PreintegratedCombinedMeasurements* self, const gtsam::PreintegratedCombinedMeasurements& expected, double tol){return self->equals(expected, tol);}, nb::arg("expected"), nb::arg("tol"))
        .def("integrateMeasurement",[](gtsam::PreintegratedCombinedMeasurements* self, const gtsam::Vector& measuredAcc, const gtsam::Vector& measuredOmega, double deltaT){ self->integrateMeasurement(measuredAcc, measuredOmega, deltaT);}, nb::arg("measuredAcc"), nb::arg("measuredOmega"), nb::arg("deltaT"))
        .def("resetIntegration",[](gtsam::PreintegratedCombinedMeasurements* self){ self->resetIntegration();})
        .def("resetIntegrationAndSetBias",[](gtsam::PreintegratedCombinedMeasurements* self, const gtsam::imuBias::ConstantBias& biasHat){ self->resetIntegrationAndSetBias(biasHat);}, nb::arg("biasHat"))
        .def("preintMeasCov",[](gtsam::PreintegratedCombinedMeasurements* self){return self->preintMeasCov();})
        .def("deltaTij",[](gtsam::PreintegratedCombinedMeasurements* self){return self->deltaTij();})
        .def("deltaRij",[](gtsam::PreintegratedCombinedMeasurements* self){return self->deltaRij();})
        .def("deltaPij",[](gtsam::PreintegratedCombinedMeasurements* self){return self->deltaPij();})
        .def("deltaVij",[](gtsam::PreintegratedCombinedMeasurements* self){return self->deltaVij();})
        .def("biasHat",[](gtsam::PreintegratedCombinedMeasurements* self){return self->biasHat();})
        .def("biasHatVector",[](gtsam::PreintegratedCombinedMeasurements* self){return self->biasHatVector();})
        .def("preintegrated", [](gtsam::PreintegratedCombinedMeasurements* self){ return self->preintegrated(); })
        .def("biasCorrectedDelta", [](gtsam::PreintegratedCombinedMeasurements* self, const gtsam::imuBias::ConstantBias& bias){return self->biasCorrectedDelta(bias);}, nb::arg("bias"))
        .def("preintegrated_H_biasAcc", [](gtsam::PreintegratedCombinedMeasurements* self) { return self->preintegrated_H_biasAcc(); })
        .def("preintegrated_H_biasOmega", [](gtsam::PreintegratedCombinedMeasurements* self) { return self->preintegrated_H_biasOmega(); })
        .def("predict",[](gtsam::PreintegratedCombinedMeasurements* self, const gtsam::NavState& state_i, const gtsam::imuBias::ConstantBias& bias){return self->predict(state_i, bias);}, nb::arg("state_i"), nb::arg("bias"));

    // TODO: add gtsam::NonlinearFactor wrapper here
    nb::class_<gtsam::CombinedImuFactor>(m_, "CombinedImuFactor")
        .def(nb::init<size_t, size_t, size_t, size_t, size_t, size_t, const gtsam::PreintegratedCombinedMeasurements&>(), nb::arg("pose_i"), nb::arg("vel_i"), nb::arg("pose_j"), nb::arg("vel_j"), nb::arg("bias_i"), nb::arg("bias_j"), nb::arg("CombinedPreintegratedMeasurements"))
        .def("preintegratedMeasurements",[](gtsam::CombinedImuFactor* self){return self->preintegratedMeasurements();})
        .def("evaluateError",[](gtsam::CombinedImuFactor* self, const gtsam::Pose3& pose_i, const gtsam::Vector& vel_i, const gtsam::Pose3& pose_j, const gtsam::Vector& vel_j, const gtsam::imuBias::ConstantBias& bias_i, const gtsam::imuBias::ConstantBias& bias_j){return self->evaluateError(pose_i, vel_i, pose_j, vel_j, bias_i, bias_j);}, nb::arg("pose_i"), nb::arg("vel_i"), nb::arg("pose_j"), nb::arg("vel_j"), nb::arg("bias_i"), nb::arg("bias_j"));


}
// clang-format on