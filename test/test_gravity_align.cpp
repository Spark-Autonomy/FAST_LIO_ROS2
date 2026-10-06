// Tests the gravity-aligned start of the world frame (gravity_align.hpp, anvil #975 plan commit 4).
#include <gtest/gtest.h>

#include <cmath>

#include "gravity_align.hpp"

namespace
{
constexpr double kDeg = M_PI / 180.0;
constexpr double kG = 9.81;  // G_m_s2 of common_lib.h, which IMU_init passes

// Rotation of the IMU in a z-up world, URDF convention: R = Rz(yaw) Ry(pitch) Rx(roll).
Eigen::Matrix3d rpy(double roll, double pitch, double yaw)
{
  return (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
      .toRotationMatrix();
}

// The accelerometer reading at rest, in g: the up vector in the IMU frame.
Eigen::Vector3d at_rest(const Eigen::Matrix3d &R) { return R.transpose() * Eigen::Vector3d::UnitZ(); }

struct Rpy
{
  double roll, pitch, yaw;
};

Rpy to_rpy(const Eigen::Matrix3d &R)
{
  return {std::atan2(R(2, 1), R(2, 2)), std::asin(-R(2, 0)), std::atan2(R(1, 0), R(0, 0))};
}

// Wraps an angle difference into (-pi, pi], so roll 180 and roll -180 compare equal.
double wrap(double a) { return std::atan2(std::sin(a), std::cos(a)); }

void expect_start(const Eigen::Vector3d &mean_acc, double roll, double pitch)
{
  state_ikfom state;
  gravity_align::set_start(state, mean_acc, kG);

  const Eigen::Matrix3d R = state.rot.toRotationMatrix();
  EXPECT_NEAR((R.transpose() * R - Eigen::Matrix3d::Identity()).norm(), 0.0, 1e-12);
  const Rpy got = to_rpy(R);
  EXPECT_NEAR(wrap(got.roll - roll), 0.0, 1e-9);
  EXPECT_NEAR(got.pitch, pitch, 1e-9);
  EXPECT_NEAR(got.yaw, 0.0, 1e-9);

  // The start rotation takes the measured up vector onto world +z.
  const Eigen::Vector3d up = R * mean_acc.normalized();
  EXPECT_NEAR(up.z(), 1.0, 1e-10);

  // Gravity points along world -z. S2 scales it to its own length, 98090 / 10000.
  const Eigen::Vector3d g = state.grav.get_vect();
  EXPECT_NEAR(g.x(), 0.0, 1e-12);
  EXPECT_NEAR(g.y(), 0.0, 1e-12);
  EXPECT_NEAR(g.z(), -9.809, 1e-12);
  EXPECT_NEAR(g.z(), -kG, 0.002);
}

TEST(GravityAlign, LevelImuStartsLevel)
{
  expect_start(Eigen::Vector3d(0.0, 0.0, 1.0), 0.0, 0.0);
}

TEST(GravityAlign, InvertedImuStartsAtRoll180)
{
  // -1 g on z: FromTwoVectors is at its antiparallel case and picks an arbitrary axis and yaw.
  expect_start(Eigen::Vector3d(0.0, 0.0, -1.0), M_PI, 0.0);
}

TEST(GravityAlign, TwoDegreeTiltIsRecovered)
{
  expect_start(at_rest(rpy(2.0 * kDeg, 0.0, 0.0)), 2.0 * kDeg, 0.0);
  expect_start(at_rest(rpy(0.0, 2.0 * kDeg, 0.0)), 0.0, 2.0 * kDeg);
  expect_start(at_rest(rpy(-1.2 * kDeg, 2.0 * kDeg, 0.0)), -1.2 * kDeg, 2.0 * kDeg);
}

TEST(GravityAlign, CabFrontMountIsRecovered)
{
  // The cab_front IMU link of tractor.xacro on a level tractor: roll 180.284 deg, pitch -0.338 deg. A real
  // heading does not change the start, because the yaw is not observed and starts at zero.
  const double roll = 3.146549, pitch = -0.005899;
  expect_start(at_rest(rpy(roll, pitch, 0.0)), roll, pitch);
  expect_start(at_rest(rpy(roll, pitch, 1.3)), roll, pitch);
  // The magnitude of the reading does not matter: the driver reports g, not m/s^2.
  expect_start(at_rest(rpy(roll, pitch, 0.0)) * 0.98, roll, pitch);
}

}  // namespace
