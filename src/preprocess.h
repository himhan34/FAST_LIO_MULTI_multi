#include <ros/ros.h> // ROS 헤더 파일 포함
#include <pcl_conversions/pcl_conversions.h> // PCL 변환 헤더 포함
#include <sensor_msgs/PointCloud2.h> // PointCloud2 메시지 헤더 포함
#include <livox_ros_driver/CustomMsg.h> // Livox ROS 드라이버의 사용자 정의 메시지 헤더 포함

using namespace std; // 표준 네임스페이스 사용

#define IS_VALID(a)  ((abs(a)>1e8) ? true : false) // 유효성 검사 매크로 정의

typedef pcl::PointXYZINormal PointType; // PCL의 포인트 타입 정의
typedef pcl::PointCloud<PointType> PointCloudXYZI; // 포인트 클라우드 타입 정의

enum LID_TYPE{AVIA = 1, VELO16, OUST64}; // LiDAR 타입 열거형 정의
enum TIME_UNIT{SEC = 0, MS = 1, US = 2, NS = 3}; // 시간 단위 열거형 정의
enum Surround{Prev, Next}; // 주변 포인트 열거형 정의
enum E_jump{Nr_nor, Nr_zero, Nr_180, Nr_inf, Nr_blind}; // 점프 상태 열거형 정의

struct orgtype // 포인트 구조체 정의
{
    double range; // 거리
    double dista; // 거리
    double angle[2]; // 각도 배열
    double intersect; // 교차점
    E_jump edj[2]; // 점프 상태 배열
    orgtype() // 기본 생성자
    {
        range = 0; // 거리 초기화
        edj[Prev] = Nr_nor; // 이전 상태 초기화
        edj[Next] = Nr_nor; // 다음 상태 초기화
        intersect = 2; // 교차점 초기화
    }
};

namespace velodyne_ros { // velodyne_ros 네임스페이스 정의
    struct EIGEN_ALIGN16 Point { // 포인트 구조체 정의
        PCL_ADD_POINT4D; // PCL 포인트 추가 매크로
        float intensity; // 강도
        float time; // 시간
        uint16_t ring; // 링 번호
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW // 정렬된 메모리 할당자 매크로
    };
}  // namespace velodyne_ros

POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point, // velodyne 포인트 등록
    (float, x, x) // X 좌표
    (float, y, y) // Y 좌표
    (float, z, z) // Z 좌표
    (float, intensity, intensity) // 강도
    (float, time, time) // 시간
    (uint16_t, ring, ring) // 링 번호
)

namespace ouster_ros { // ouster_ros 네임스페이스 정의
    struct EIGEN_ALIGN16 Point { // 포인트 구조체 정의
        PCL_ADD_POINT4D; // PCL 포인트 추가 매크로
        float intensity; // 강도
        uint32_t t; // 타임스탬프
        uint16_t reflectivity; // 반사율
        uint8_t ring; // 링 번호
        uint16_t ambient; // 주변 광량
        uint32_t range; // 거리
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW // 정렬된 메모리 할당자 매크로
    };
}  // namespace ouster_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point, // ouster 포인트 등록
    (float, x, x) // X 좌표
    (float, y, y) // Y 좌표
    (float, z, z) // Z 좌표
    (float, intensity, intensity) // 강도
    // PCL::uint32_t를 피하기 위해 std::uint32_t 사용
    (std::uint32_t, t, t) // 타임스탬프
    (std::uint16_t, reflectivity, reflectivity) // 반사율
    (std::uint8_t, ring, ring) // 링 번호
    (std::uint16_t, ambient, ambient) // 주변 광량
    (std::uint32_t, range, range) // 거리
)

class Preprocess // Preprocess 클래스 정의
{
  public:
  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW // 정렬된 메모리 할당자 매크로 (주석 처리됨)

  Preprocess(); // 생성자
  ~Preprocess(); // 소멸자
  
  void process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out, const int &lidar_num); // Livox 메시지 처리 함수
  void process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out, const int &lidar_num); // 일반 메시지 처리 함수
  void set(); // 설정 함수

  PointCloudXYZI pl_full, pl_surf; // 포인트 클라우드 변수
  PointCloudXYZI pl_buff[128]; // 최대 128개의 라이다 포인트 클라우드 배열
  vector<orgtype> typess[128]; // 최대 128개의 라이다 유형 배열
  float time_unit_scale[2]; // 시간 단위 스케일 배열
  int lidar_type[2], point_filter_num[2], N_SCANS[2], SCAN_RATE[2], time_unit[2]; // 라이다 타입, 포인트 필터 수, 스캔 수, 스캔 속도, 시간 단위 배열
  double blind[2]; // 블라인드 값 배열
  bool given_offset_time; // 오프셋 시간이 주어졌는지 여부

  private:
  void avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg, const int &lidar_num); // AVIA 라이다 처리 함수
  void oust64_handler(const sensor_msgs::PointCloud2::ConstPtr &msg, const int &lidar_num); // OUST64 라이다 처리 함수
  void velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg, const int &lidar_num); // VELODYNE 라이다 처리 함수
};
