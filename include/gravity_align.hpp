// Gravity-aligned start of the world frame (anvil #975, plan commit 4).
//
// Upstream keeps the initial IMU rotation at identity and points gravity along the mean accelerometer vector, so
// the world frame is the first IMU pose and the published roll and pitch are relative to the start attitude. This
// start rotates the IMU so that the mean accelerometer vector points along +z and sets gravity to (0, 0, -g). The
// world z axis is then up, and the published roll and pitch are absolute. The initial yaw is zero.
#pragma once

#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "use-ikfom.hpp"

namespace gravity_align
{

// The initial rotation of the IMU in the world frame, from the mean accelerometer vector at rest.
//
// At rest the accelerometer reads the specific force, which points up. FromTwoVectors gives the rotation that maps
// it onto +z, and it is well defined near 180 deg, the case of an inverted IMU. The cross-product form that
// upstream left commented out in IMU_init has no defined axis there. FromTwoVectors picks an arbitrary yaw near
// 180 deg, so the yaw is removed afterwards. Roll and pitch depend only on R^T z, which a yaw does not change.
// Not defined at a pitch of +-90 deg.
inline Eigen::Matrix3d initial_rotation(const Eigen::Vector3d &mean_acc)
{
  // Near 180 deg the quaternion norm is off by about 1e-11, from 1 + cos(angle). Normalize it, because the filter
  // state stores the result as a quaternion.
  const Eigen::Matrix3d R =
      Eigen::Quaterniond::FromTwoVectors(mean_acc, Eigen::Vector3d::UnitZ()).normalized().toRotationMatrix();
  const double yaw = std::atan2(R(1, 0), R(0, 0));
  return Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() * R;
}

// Sets the start rotation and the gravity of the filter state. S2 scales gravity to its own length, 9.809.
inline void set_start(state_ikfom &state, const Eigen::Vector3d &mean_acc, double g)
{
  state.rot = SO3(Eigen::Quaterniond(initial_rotation(mean_acc)).normalized());
  state.grav = S2(Eigen::Vector3d(0.0, 0.0, -g));
}

}  // namespace gravity_align
