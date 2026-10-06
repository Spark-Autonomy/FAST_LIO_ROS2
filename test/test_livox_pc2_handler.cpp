// Tests the lidar_type 5 handler (Preprocess::livox_pc2_handler) on the livox_ros_driver2 PointCloud2 layout.
#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include "preprocess.h"

namespace
{
using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

constexpr uint64_t kBaseNs = 1759700000123456789ULL;  // header stamp, ns

struct FixturePoint
{
  float x, y, z, intensity;
  uint8_t tag, line;
  double offset_ms;  // point time after the header stamp
};

PointField field(const char *name, uint32_t offset, uint8_t datatype)
{
  PointField f;
  f.name = name;
  f.offset = offset;
  f.datatype = datatype;
  f.count = 1;
  return f;
}

// The layout of lddc.cpp InitPointcloud2Msg at livox_ros_driver2 5cbae244: LivoxPointXyzrtlt, 26 bytes a point.
std::unique_ptr<PointCloud2> livox_cloud(const std::vector<FixturePoint> &pts)
{
  auto msg = std::make_unique<PointCloud2>();
  msg->header.stamp.sec = int32_t(kBaseNs / 1000000000ULL);
  msg->header.stamp.nanosec = uint32_t(kBaseNs % 1000000000ULL);
  msg->height = 1;
  msg->width = uint32_t(pts.size());
  msg->fields = {field("x", 0, PointField::FLOAT32),          field("y", 4, PointField::FLOAT32),
                 field("z", 8, PointField::FLOAT32),          field("intensity", 12, PointField::FLOAT32),
                 field("tag", 16, PointField::UINT8),         field("line", 17, PointField::UINT8),
                 field("timestamp", 18, PointField::FLOAT64)};
  msg->point_step = 26;
  msg->row_step = msg->width * msg->point_step;
  msg->is_bigendian = false;
  msg->is_dense = true;
  msg->data.resize(msg->row_step);
  for (size_t i = 0; i < pts.size(); ++i)
  {
    uint8_t *p = msg->data.data() + i * 26;
    const auto &q = pts[i];
    const double t_ns = double(kBaseNs) + q.offset_ms * 1e6;  // the driver casts an absolute uint64 ns to double
    memcpy(p + 0, &q.x, 4);
    memcpy(p + 4, &q.y, 4);
    memcpy(p + 8, &q.z, 4);
    memcpy(p + 12, &q.intensity, 4);
    p[16] = q.tag;
    p[17] = q.line;
    memcpy(p + 18, &t_ns, 8);
  }
  return msg;
}

// N points at 2 to 11 m on lines 0 to 3, 0.1 ms apart. The latest point is in the middle, as the driver order is
// not strictly by time.
std::vector<FixturePoint> valid_points(size_t n)
{
  std::vector<FixturePoint> pts;
  for (size_t i = 0; i < n; ++i)
    pts.push_back({2.0f + float(i % 10), 0.5f, 0.1f, float(i), 0x00, uint8_t(i % 4), 0.1 * double(i)});
  std::swap(pts[n / 2].offset_ms, pts[n - 1].offset_ms);
  return pts;
}

class LivoxPc2Handler : public ::testing::Test
{
protected:
  void SetUp() override
  {
    pre.set(false, LIVOX_PC2, 1.0, 1);
    pre.N_SCANS = 4;
    pre.time_unit = NS;
    pre.max_range = 20.0;
  }

  PointCloudXYZI::Ptr run(std::unique_ptr<PointCloud2> msg)
  {
    PointCloudXYZI::Ptr out(new PointCloudXYZI());
    pre.process(msg, out);
    return out;
  }

  Preprocess pre;
};

TEST_F(LivoxPc2Handler, DriverLayoutYieldsAllPointsWithTheirTimes)
{
  const size_t n = 50;
  const auto pts = valid_points(n);
  const auto out = run(livox_cloud(pts));

  ASSERT_EQ(out->size(), n);
  std::vector<double> want, got;
  for (const auto &q : pts)
    want.push_back(q.offset_ms);
  for (const auto &p : out->points)
    got.push_back(p.curvature);
  std::sort(want.begin(), want.end());
  std::sort(got.begin(), got.end());
  for (size_t i = 0; i < n; ++i)
    EXPECT_NEAR(got[i], want[i], 1e-3) << "point " << i;  // 1 us; the double ns carries 256 ns resolution

  // sync_packages reads points.back() as the scan end time.
  EXPECT_NEAR(out->points.back().curvature, 0.1 * double(n - 1), 1e-3);
  for (const auto &p : out->points)
    EXPECT_LE(p.curvature, out->points.back().curvature);
}

TEST_F(LivoxPc2Handler, PointFieldsAreCopied)
{
  const auto out = run(livox_cloud({{3.0f, -4.0f, 1.5f, 77.0f, 0x10, 2, 0.25}}));
  ASSERT_EQ(out->size(), 1u);
  const auto &p = out->points[0];
  EXPECT_FLOAT_EQ(p.x, 3.0f);
  EXPECT_FLOAT_EQ(p.y, -4.0f);
  EXPECT_FLOAT_EQ(p.z, 1.5f);
  EXPECT_FLOAT_EQ(p.intensity, 77.0f);
  EXPECT_NEAR(p.curvature, 0.25, 1e-3);
}

TEST_F(LivoxPc2Handler, RangeLineAndTagFilters)
{
  const auto out = run(livox_cloud({
      {5.0f, 0, 0, 0, 0x00, 0, 0.0},   // kept
      {0.5f, 0, 0, 0, 0x00, 0, 0.1},   // inside blind 1.0 m
      {25.0f, 0, 0, 0, 0x00, 0, 0.2},  // beyond max_range 20 m
      {5.0f, 0, 0, 0, 0x00, 4, 0.3},   // line >= N_SCANS
      {5.0f, 0, 0, 0, 0x20, 0, 0.4},   // tag return bits 0x20
      {6.0f, 0, 0, 0, 0x10, 3, 0.5},   // kept
  }));
  ASSERT_EQ(out->size(), 2u);
  EXPECT_FLOAT_EQ(out->points[0].x, 5.0f);
  EXPECT_FLOAT_EQ(out->points[1].x, 6.0f);
}

TEST_F(LivoxPc2Handler, MaxRangeZeroDisablesTheLimit)
{
  pre.max_range = 0.0;
  EXPECT_EQ(run(livox_cloud({{250.0f, 0, 0, 0, 0x00, 0, 0.0}}))->size(), 1u);
}

TEST_F(LivoxPc2Handler, PointFilterNumKeepsEveryNthValidPoint)
{
  pre.point_filter_num = 3;
  EXPECT_EQ(run(livox_cloud(valid_points(30)))->size(), 10u);
}

// Each case changes one property of the driver layout. Every case must yield a dropped cloud.
struct BadLayout
{
  const char *name;
  void (*mutate)(PointCloud2 &);
};

const BadLayout kBadLayouts[] = {
    {"point_step 32",
     [](PointCloud2 &m) {
       m.point_step = 32;
       m.row_step = m.width * 32;
       m.data.resize(m.row_step);
     }},
    {"timestamp offset 20, same point_step", [](PointCloud2 &m) { m.fields[6].offset = 20; }},
    {"timestamp FLOAT32", [](PointCloud2 &m) { m.fields[6].datatype = PointField::FLOAT32; }},
    {"intensity renamed reflectivity", [](PointCloud2 &m) { m.fields[3].name = "reflectivity"; }},
    {"field count 2", [](PointCloud2 &m) { m.fields[0].count = 2; }},
    {"no timestamp field", [](PointCloud2 &m) { m.fields.pop_back(); }},
    {"extra field", [](PointCloud2 &m) { m.fields.push_back(field("ring", 25, PointField::UINT8)); }},
    {"big endian", [](PointCloud2 &m) { m.is_bigendian = true; }},
    {"row_step padded", [](PointCloud2 &m) { m.row_step += 2; }},
    {"data short by one byte", [](PointCloud2 &m) { m.data.pop_back(); }},
};

TEST_F(LivoxPc2Handler, AnyOtherLayoutYieldsADroppedCloud)
{
  for (const auto &bad : kBadLayouts)
  {
    // A good cloud first: a layout result from an earlier cloud must not carry over.
    ASSERT_EQ(run(livox_cloud(valid_points(10)))->size(), 10u) << bad.name;
    auto msg = livox_cloud(valid_points(10));
    bad.mutate(*msg);
    EXPECT_EQ(run(std::move(msg))->size(), 0u) << bad.name;
  }
  // The handler recovers when the driver layout returns.
  EXPECT_EQ(run(livox_cloud(valid_points(10)))->size(), 10u);
}

TEST_F(LivoxPc2Handler, EmptyCloudYieldsNoPoints)
{
  EXPECT_EQ(run(livox_cloud({}))->size(), 0u);
}

}  // namespace
