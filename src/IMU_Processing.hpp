#include <cmath>  // 수학 함수들을 사용하기 위한 헤더 파일 포함
#include <deque>  // 덱 자료 구조를 사용하기 위한 헤더 파일 포함
#include <csignal>  // 시그널 처리 기능을 제공하는 헤더 파일 포함
#include <Eigen/Eigen>  // Eigen 라이브러리에서 선형대수 관련 기능을 사용하기 위한 헤더 파일 포함
#include <pcl/point_cloud.h>  // PCL(Point Cloud Library)에서 포인트 클라우드를 사용하기 위한 헤더 파일 포함
#include <pcl/point_types.h>  // PCL에서 제공하는 다양한 포인트 타입을 사용하기 위한 헤더 파일 포함
#include <pcl_conversions/pcl_conversions.h>  // PCL 포인트 클라우드와 ROS 메시지 사이의 변환을 위한 헤더 파일 포함
#include <ros/ros.h>  // ROS 기능을 사용하기 위한 헤더 파일 포함
#include <sensor_msgs/Imu.h>  // ROS에서 IMU(Inertial Measurement Unit) 센서 데이터를 처리하기 위한 메시지 타입 포함
#include "so3_math.h"  // 회전 관련 수학적 연산을 지원하는 SO3 관련 사용자 정의 헤더 파일 포함
#include "common_lib.h"  // 공통적으로 사용하는 함수나 데이터 구조들을 포함하는 사용자 정의 헤더 파일 포함
#include "use-ikfom.hpp"  // IKFoM(Incremental Kalman Filter on Manifolds) 구현을 포함하는 사용자 정의 헤더 파일 포함

/// *************Preconfiguration



#define MAX_INI_COUNT (10)  // 초기화 카운트의 최대값을 10으로 정의

const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};  
// 포인트 타입의 곡률 값을 비교하는 함수, x의 곡률이 y의 곡률보다 작은지 여부를 반환

/// ************* IMU 처리 및 왜곡 보정 클래스
class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // Eigen에서 메모리 정렬을 위해 사용하는 매크로
  
  ImuProcess();  // 생성자 선언
  ~ImuProcess();  // 소멸자 선언
  
  void Reset();  // 초기화 함수 선언
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);  // IMU 데이터를 사용한 초기화 함수
  void set_extrinsic(const V3D &transl, const M3D &rot);  // 외부 파라미터(변환 및 회전 행렬)를 설정하는 함수
  void set_extrinsic(const V3D &transl);  // 변환 값만을 사용하는 외부 파라미터 설정 함수
  void set_extrinsic(const MD(4,4) &T);  // 4x4 변환 행렬로 외부 파라미터를 설정하는 함수
  void set_gyr_cov(const V3D &scaler);  // 자이로스코프 공분산을 설정하는 함수
  void set_acc_cov(const V3D &scaler);  // 가속도계 공분산을 설정하는 함수
  void set_gyr_bias_cov(const V3D &b_g);  // 자이로스코프 바이어스 공분산을 설정하는 함수
  void set_acc_bias_cov(const V3D &b_a);  // 가속도계 바이어스 공분산을 설정하는 함수
  Eigen::Matrix<double, 12, 12> Q;  // 12x12 크기의 공분산 행렬 Q 정의
  void Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr pcl_un_, const bool &multi_lidar, int lidar_num = 1);  
  // IMU 측정 데이터를 처리하고, 상태를 업데이트하는 함수
  
  V3D cov_acc;  // 가속도계 공분산 벡터
  V3D cov_gyr;  // 자이로스코프 공분산 벡터
  V3D cov_acc_scale;  // 가속도계 스케일링 공분산 벡터
  V3D cov_gyr_scale;  // 자이로스코프 스케일링 공분산 벡터
  V3D cov_bias_gyr;  // 자이로스코프 바이어스 공분산 벡터
  V3D cov_bias_acc;  // 가속도계 바이어스 공분산 벡터

 private:
  void IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);  
  // IMU 초기화를 수행하는 함수

  void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_in_out, int lidar_num = 1);  
  // 단일 LiDAR에서 받은 포인트 클라우드의 왜곡을 보정하는 함수

  void UndistortPclMultiLiDAR(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_in_out);  
  // 다중 LiDAR에서 받은 포인트 클라우드의 왜곡을 보정하는 함수

  PointCloudXYZI::Ptr cur_pcl_un_;  // 현재 처리 중인 포인트 클라우드를 저장하는 포인터
  sensor_msgs::ImuConstPtr last_imu_;  // 마지막으로 수신된 IMU 데이터를 저장하는 포인터
  deque<sensor_msgs::ImuConstPtr> v_imu_;  // IMU 데이터를 저장하는 덱 자료구조
  vector<Pose6D> IMUpose;  // IMU의 포즈(자세)를 저장하는 벡터
  vector<M3D>    v_rot_pcl_;  // 포인트 클라우드의 회전 행렬을 저장하는 벡터
  M3D Lidar_R_wrt_IMU;  // LiDAR에 대한 IMU의 회전 행렬
  V3D Lidar_T_wrt_IMU;  // LiDAR에 대한 IMU의 변환 벡터
  V3D mean_acc;  // 가속도 평균 값
  V3D mean_gyr;  // 자이로스코프 평균 값
  V3D angvel_last;  // 마지막 각속도 값
  V3D acc_s_last;  // 마지막 가속도 값
  double start_timestamp_;  // 시작 시점의 타임스탬프
  double last_lidar_end_time_, last_lidar_end_time_2;  // 마지막 LiDAR 데이터의 종료 시점 타임스탬프
  int    init_iter_num = 1;  // 초기화 반복 횟수
  bool   b_first_frame_ = true;  // 첫 프레임인지 여부를 나타내는 플래그
  bool   imu_need_init_ = true;  // IMU 초기화가 필요한지 여부를 나타내는 플래그
};


ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1)
{
  // 생성자: 클래스 변수들을 초기화
  init_iter_num = 1;  // 초기화 반복 횟수를 1로 설정
  Q = process_noise_cov();  // 프로세스 노이즈 공분산을 Q로 설정
  cov_acc       = V3D(0.1, 0.1, 0.1);  // 가속도계의 공분산 값 설정
  cov_gyr       = V3D(0.1, 0.1, 0.1);  // 자이로스코프의 공분산 값 설정
  cov_bias_gyr  = V3D(0.0001, 0.0001, 0.0001);  // 자이로스코프 바이어스 공분산 값 설정
  cov_bias_acc  = V3D(0.0001, 0.0001, 0.0001);  // 가속도계 바이어스 공분산 값 설정
  mean_acc      = V3D(0, 0, -1.0);  // 가속도의 평균 값을 설정 (-1.0은 중력 가속도를 의미)
  mean_gyr      = V3D(0, 0, 0);  // 자이로스코프의 평균 값을 설정
  angvel_last     = Zero3d;  // 마지막 각속도를 0으로 초기화
  Lidar_T_wrt_IMU = Zero3d;  // LiDAR에 대한 IMU 변환을 0으로 초기화
  Lidar_R_wrt_IMU = Eye3d;  // LiDAR에 대한 IMU 회전을 단위 행렬로 초기화
  last_imu_.reset(new sensor_msgs::Imu());  // 마지막 IMU 데이터를 새로운 IMU 메시지로 초기화
}

ImuProcess::~ImuProcess() {}  // 소멸자: 별도의 소멸 작업 없음

void ImuProcess::Reset() 
{
  // ROS_WARN("Reset ImuProcess");  // IMU 프로세스를 리셋할 때 경고 메시지를 출력하는 부분(주석 처리됨)
  
  mean_acc      = V3D(0, 0, -1.0);  // 가속도의 평균 값을 다시 초기화 (-1.0은 중력 가속도를 의미)
  mean_gyr      = V3D(0, 0, 0);  // 자이로스코프의 평균 값을 다시 0으로 초기화
  angvel_last       = Zero3d;  // 마지막 각속도 값을 0으로 초기화
  imu_need_init_    = true;  // IMU가 초기화가 필요함을 true로 설정
  start_timestamp_  = -1;  // 시작 타임스탬프를 -1로 초기화
  init_iter_num     = 1;  // 초기화 반복 횟수를 1로 설정
  v_imu_.clear();  // IMU 데이터를 저장하는 덱을 초기화
  IMUpose.clear();  // IMU 포즈 벡터를 초기화
  last_imu_.reset(new sensor_msgs::Imu());  // 마지막 IMU 데이터를 새로운 IMU 메시지로 초기화
  cur_pcl_un_.reset(new PointCloudXYZI());  // 현재 포인트 클라우드를 새로운 포인트 클라우드로 초기화
}


void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
  Lidar_T_wrt_IMU = transl;  // LiDAR에 대한 IMU의 변환 벡터 설정
  Lidar_R_wrt_IMU = rot;  // LiDAR에 대한 IMU의 회전 행렬 설정
}

void ImuProcess::set_gyr_cov(const V3D &scaler)
{
  cov_gyr_scale = scaler;  // 자이로스코프 공분산 스케일 설정
}

void ImuProcess::set_acc_cov(const V3D &scaler)
{
  cov_acc_scale = scaler;  // 가속도계 공분산 스케일 설정
}

void ImuProcess::set_gyr_bias_cov(const V3D &b_g)
{
  cov_bias_gyr = b_g;  // 자이로스코프 바이어스 공분산 설정
}

void ImuProcess::set_acc_bias_cov(const V3D &b_a)
{
  cov_bias_acc = b_a;  // 가속도계 바이어스 공분산 설정
}

void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N)
{
  /** 1. 중력, 자이로 바이어스, 가속도 및 자이로 공분산 초기화
   ** 2. 가속도 측정을 단위 중력으로 정규화 **/

  V3D cur_acc, cur_gyr;  // 현재 가속도와 자이로스코프 데이터를 저장할 변수

  if (b_first_frame_)  // 첫 프레임일 경우
  {
    Reset();  // 초기화 함수 호출
    N = 1;  // 데이터 카운터 N을 1로 설정
    b_first_frame_ = false;  // 첫 프레임 플래그를 false로 설정
    const auto &imu_acc = meas.imu.front()->linear_acceleration;  // 첫 IMU 데이터에서 가속도 값을 가져옴
    const auto &gyr_acc = meas.imu.front()->angular_velocity;  // 첫 IMU 데이터에서 각속도 값을 가져옴
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;  // 가속도 평균값에 첫 가속도 값을 설정
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;  // 자이로스코프 평균값에 첫 자이로 값을 설정
  }

  for (const auto &imu : meas.imu)  // IMU 데이터를 반복하면서 처리
  {
    const auto &imu_acc = imu->linear_acceleration;  // 현재 IMU 가속도 값을 가져옴
    const auto &gyr_acc = imu->angular_velocity;  // 현재 IMU 각속도 값을 가져옴
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;  // 현재 가속도 값을 벡터로 설정
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;  // 현재 자이로 값을 벡터로 설정

    mean_acc      += (cur_acc - mean_acc) / N;  // 가속도 평균을 업데이트
    mean_gyr      += (cur_gyr - mean_gyr) / N;  // 자이로스코프 평균을 업데이트

    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
    // 가속도 공분산을 업데이트

    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);
    // 자이로스코프 공분산을 업데이트

    // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;  // 가속도 벡터의 노름 출력(주석 처리됨)

    N ++;  // N을 증가시켜 다음 데이터를 처리할 준비
  }

  state_ikfom init_state = kf_state.get_x();  // 현재 상태를 가져옴
  init_state.grav = S2(- mean_acc / mean_acc.norm() * G_m_s2);  // 중력 벡터를 평균 가속도를 기준으로 설정
  
  // state_inout.rot = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));  // 회전 값을 설정하는 부분(주석 처리됨)
  init_state.bg  = mean_gyr;  // 자이로 바이어스를 평균 자이로 값으로 설정
  init_state.offset_T_L_I = Lidar_T_wrt_IMU;  // LiDAR에 대한 IMU의 변환 값을 설정
  init_state.offset_R_L_I = Lidar_R_wrt_IMU;  // LiDAR에 대한 IMU의 회전 값을 설정
  kf_state.change_x(init_state);  // 상태를 업데이트

  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();  // 상태 공분산 행렬을 가져옴
  init_P.setIdentity();  // 공분산 행렬을 단위 행렬로 설정
  init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;  // 자이로스코프 공분산 설정
  init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;  // 가속도계 공분산 설정
  init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001;  // 자이로 바이어스 공분산 설정
  init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;  // 가속도 바이어스 공분산 설정
  init_P(21,21) = init_P(22,22) = 0.00001;  // LiDAR에 대한 IMU 변환 공분산 설정
  kf_state.change_P(init_P);  // 공분산 행렬을 업데이트
  last_imu_ = meas.imu.back();  // 마지막 IMU 데이터를 업데이트
}

void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_out, int lidar_num)
{
  /*** 이전 프레임의 마지막 IMU 데이터를 현재 프레임의 첫 번째 IMU 데이터로 추가 ***/
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);  // 마지막 IMU 데이터를 앞에 추가
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();  // 첫 번째 IMU 데이터의 시작 시간
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();  // 마지막 IMU 데이터의 끝 시간
  const double &pcl_beg_time = lidar_num == 1 ? meas.lidar_beg_time : meas.lidar_beg_time2;  // LiDAR 데이터의 시작 시간
  const double &pcl_end_time = lidar_num == 1 ? meas.lidar_end_time : meas.lidar_end_time2;  // LiDAR 데이터의 끝 시간
  
  /*** 포인트 클라우드를 오프셋 시간에 따라 정렬 ***/
  pcl_out = *(meas.lidar);  // LiDAR 데이터를 복사
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);  // 포인트 클라우드를 시간에 따라 정렬
  // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;  // 디버깅용 출력 (주석 처리됨)

  /*** IMU 포즈 초기화 ***/
  state_ikfom imu_state = kf_state.get_x();  // 현재 상태를 가져옴
  IMUpose.clear();  // IMU 포즈 벡터 초기화
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));  // 초기 포즈 추가

  /*** 각 IMU 데이터에서 포즈를 예측 ***/
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;  // 각속도, 가속도, 속도, 위치 변수
  M3D R_imu;  // 회전 행렬

  double dt = 0;  // 시간 차이

  input_ikfom in;  // 입력 데이터 구조체
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)  // IMU 데이터를 순차적으로 처리
  {
    auto &&head = *(it_imu);  // 현재 IMU 데이터
    auto &&tail = *(it_imu + 1);  // 다음 IMU 데이터
    
    if (tail->header.stamp.toSec() < last_lidar_end_time_)    continue;  // LiDAR 끝 시간 이전이면 건너뜀
    
    // 각속도와 가속도의 평균 계산
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    acc_avr     = acc_avr * G_m_s2 / mean_acc.norm();  // 가속도를 중력에 맞춰 정규화

    if(head->header.stamp.toSec() < last_lidar_end_time_)  // LiDAR 끝 시간 이전일 경우
    {
      dt = tail->header.stamp.toSec() - last_lidar_end_time_;  // 시간 차이 계산
      // dt = tail->header.stamp.toSec() - pcl_beg_time;  // (주석 처리됨)
    }
    else
    {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();  // IMU 데이터 간의 시간 차이 계산
    }
    
    in.acc = acc_avr;  // 가속도 값 설정
    in.gyro = angvel_avr;  // 자이로 값 설정
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;  // 자이로 공분산 설정
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;  // 가속도 공분산 설정
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;  // 자이로 바이어스 공분산 설정
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;  // 가속도 바이어스 공분산 설정
    kf_state.predict(dt, Q, in);  // 칼만 필터 예측 단계 수행

    /* 각 IMU 데이터에서 포즈 저장 */
    imu_state = kf_state.get_x();  // 상태 업데이트
    angvel_last = angvel_avr - imu_state.bg;  // 자이로 바이어스 보정
    acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba);  // 가속도 보정
    for(int i=0; i<3; i++)
    {
      acc_s_last[i] += imu_state.grav[i];  // 중력 보정
    }
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;  // 오프셋 시간 계산
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));  // 포즈 추가
  }

  /*** 프레임의 마지막에서 위치 및 자세 예측 ***/
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;  // 방향 설정
  dt = note * (pcl_end_time - imu_end_time);  // 시간 차이 계산
  kf_state.predict(dt, Q, in);  // 마지막 예측 단계 수행
  
  imu_state = kf_state.get_x();  // 상태 업데이트
  last_imu_ = meas.imu.back();  // 마지막 IMU 데이터 업데이트
  last_lidar_end_time_ = pcl_end_time;  // LiDAR 끝 시간 업데이트

  /*** 각 LiDAR 포인트의 왜곡 보정 (역방향 처리) ***/
  if (pcl_out.points.begin() == pcl_out.points.end()) return;  // 포인트가 없으면 종료
  auto it_pcl = pcl_out.points.end() - 1;  // 마지막 포인트부터 시작
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
  {
    auto head = it_kp - 1;  // 현재 포즈
    auto tail = it_kp;  // 다음 포즈
    R_imu<<MAT_FROM_ARRAY(head->rot);  // 회전 행렬 설정
    vel_imu<<VEC_FROM_ARRAY(head->vel);  // 속도 설정
    pos_imu<<VEC_FROM_ARRAY(head->pos);  // 위치 설정
    acc_imu<<VEC_FROM_ARRAY(tail->acc);  // 가속도 설정
    angvel_avr<<VEC_FROM_ARRAY(tail->gyr);  // 자이로 값 설정

    for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)  // 포인트 클라우드 처리
    {
      dt = it_pcl->curvature / double(1000) - head->offset_time;  // 시간 차이 계산

      /* 포인트를 'end' 프레임으로 변환, 회전만 사용
       * 보정 방향은 프레임 이동 방향과 반대
       * 보정된 포인트 P_compensate 계산 */
      M3D R_i(R_imu * Exp(angvel_avr, dt));  // 회전 행렬 계산
      
      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);  // 포인트 좌표
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);  // 변환 벡터 계산
      V3D P_compensate = imu_state.offset_R_L_I.conjugate() * (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);  // 보정된 포인트 계산 (정확하지 않음!)
      
      // 보정된 포인트 좌표 저장
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;  // 첫 번째 포인트까지 처리하면 종료
    }
  }
}

void ImuProcess::UndistortPclMultiLiDAR(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_out)
{
  /*** 이전 프레임의 마지막 IMU 데이터를 현재 프레임의 첫 번째 IMU 데이터로 추가 ***/
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);  // 마지막 IMU 데이터를 앞에 추가
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();  // 첫 번째 IMU 데이터의 시작 시간
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();  // 마지막 IMU 데이터의 끝 시간
  const double &pcl_beg_time = meas.lidar_beg_time < meas.lidar_beg_time2 ? meas.lidar_beg_time : meas.lidar_beg_time2;  // LiDAR 데이터의 시작 시간
  const double &pcl_end_time = meas.lidar_end_time > meas.lidar_end_time2 ? meas.lidar_end_time : meas.lidar_end_time2;  // LiDAR 데이터의 끝 시간

  /*** 포인트 클라우드를 오프셋 시간에 따라 정렬 ***/
  PointCloudXYZI pcl_1_out = *(meas.lidar);  // 첫 번째 LiDAR 데이터 복사
  PointCloudXYZI pcl_2_out = *(meas.lidar2);  // 두 번째 LiDAR 데이터 복사

  sort(pcl_1_out.points.begin(), pcl_1_out.points.end(), time_list);  // 첫 번째 LiDAR 데이터를 시간에 따라 정렬
  sort(pcl_2_out.points.begin(), pcl_2_out.points.end(), time_list);  // 두 번째 LiDAR 데이터를 시간에 따라 정렬
  // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;  // 디버깅용 출력 (주석 처리됨)
  
  /*** IMU 포즈 초기화 ***/
  state_ikfom imu_state = kf_state.get_x();  // 현재 상태 가져오기
  IMUpose.clear();  // IMU 포즈 벡터 초기화
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));  // 초기 포즈 추가

  /*** 각 IMU 데이터를 통해 포즈 예측 ***/
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;  // 각속도, 가속도, 속도, 위치 변수
  M3D R_imu;  // 회전 행렬
  double dt = 0;  // 시간 차이
  input_ikfom in;  // 입력 데이터 구조체
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)  // IMU 데이터를 순차적으로 처리
  {
    auto &&head = *(it_imu);  // 현재 IMU 데이터
    auto &&tail = *(it_imu + 1);  // 다음 IMU 데이터
    
    if (tail->header.stamp.toSec() < last_lidar_end_time_)    continue;  // LiDAR 끝 시간 이전이면 건너뜀
    
    // 각속도와 가속도의 평균 계산
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    acc_avr     = acc_avr * G_m_s2 / mean_acc.norm();  // 가속도를 중력에 맞춰 정규화

    if(head->header.stamp.toSec() < last_lidar_end_time_)  // LiDAR 끝 시간 이전일 경우
    {
      dt = tail->header.stamp.toSec() - last_lidar_end_time_;  // 시간 차이 계산
    }
    else
    {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();  // IMU 데이터 간의 시간 차이 계산
    }
    
    in.acc = acc_avr;  // 가속도 값 설정
    in.gyro = angvel_avr;  // 자이로 값 설정
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;  // 자이로 공분산 설정
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;  // 가속도 공분산 설정
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;  // 자이로 바이어스 공분산 설정
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;  // 가속도 바이어스 공분산 설정
    kf_state.predict(dt, Q, in);  // 칼만 필터 예측 단계 수행

    /* 각 IMU 데이터에서 포즈 저장 */
    imu_state = kf_state.get_x();  // 상태 업데이트
    angvel_last = angvel_avr - imu_state.bg;  // 자이로 바이어스 보정
    acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba);  // 가속도 보정
    for(int i=0; i<3; i++)
    {
      acc_s_last[i] += imu_state.grav[i];  // 중력 보정
    }
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;  // 오프셋 시간 계산
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));  // 포즈 추가
  }

  /*** 프레임의 마지막에서 위치 및 자세 예측 ***/
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;  // 방향 설정
  dt = note * (pcl_end_time - imu_end_time);  // 시간 차이 계산
  kf_state.predict(dt, Q, in);  // 마지막 예측 단계 수행
  
  imu_state = kf_state.get_x();  // 상태 업데이트
  last_imu_ = meas.imu.back();  // 마지막 IMU 데이터 업데이트
  last_lidar_end_time_ = pcl_end_time;  // LiDAR 끝 시간 업데이트

  /*** 각 LiDAR 포인트의 왜곡 보정 (역방향 처리) ***/
  {
    if (pcl_1_out.points.begin() == pcl_1_out.points.end()) return;  // 첫 번째 LiDAR 데이터가 없으면 종료
    auto it_pcl = pcl_1_out.points.end() - 1;  // 마지막 포인트부터 시작
    for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)  // IMU 포즈 처리
    {
      auto head = it_kp - 1;  // 현재 포즈
      auto tail = it_kp;  // 다음 포즈
      R_imu<<MAT_FROM_ARRAY(head->rot);  // 회전 행렬 설정
      vel_imu<<VEC_FROM_ARRAY(head->vel);  // 속도 설정
      pos_imu<<VEC_FROM_ARRAY(head->pos);  // 위치 설정
      acc_imu<<VEC_FROM_ARRAY(tail->acc);  // 가속도 설정
      angvel_avr<<VEC_FROM_ARRAY(tail->gyr);  // 자이로 값 설정

      for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)  // 포인트 클라우드 처리
      {
        dt = it_pcl->curvature / double(1000) - head->offset_time;  // 시간 차이 계산

        /* 포인트를 'end' 프레임으로 변환, 회전만 사용
         * 보정된 포인트 P_compensate 계산 */
        M3D R_i(R_imu * Exp(angvel_avr, dt));  // 회전 행렬 계산
        
        V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);  // 포인트 좌표
        V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);  // 변환 벡터 계산
        V3D P_compensate = imu_state.offset_R_L_I.conjugate() * (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);  // 보정된 포인트 계산 (정확하지 않음!)
        
        // 보정된 포인트 좌표 저장
        it_pcl->x = P_compensate(0);
        it_pcl->y = P_compensate(1);
        it_pcl->z = P_compensate(2);

        if (it_pcl == pcl_1_out.points.begin()) break;  // 첫 번째 포인트까지 처리하면 종료
      }
    }
  }
  {
    if (pcl_2_out.points.begin() == pcl_2_out.points.end()) return;  // 두 번째 LiDAR 데이터가 없으면 종료
    auto it_pcl = pcl_2_out.points.end() - 1;  // 마지막 포인트부터 시작
    for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)  // IMU 포즈 처리
    {
      auto head = it_kp - 1;  // 현재 포즈
      auto tail = it_kp;  // 다음 포즈
      R_imu<<MAT_FROM_ARRAY(head->rot);  // 회전 행렬 설정
      vel_imu<<VEC_FROM_ARRAY(head->vel);  // 속도 설정
      pos_imu<<VEC_FROM_ARRAY(head->pos);  // 위치 설정
      acc_imu<<VEC_FROM_ARRAY(tail->acc);  // 가속도 설정
      angvel_avr<<VEC_FROM_ARRAY(tail->gyr);  // 자이로 값 설정

      for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)  // 포인트 클라우드 처리
      {
        dt = it_pcl->curvature / double(1000) - head->offset_time;  // 시간 차이 계산
        M3D R_i(R_imu * Exp(angvel_avr, dt));  // 회전 행렬 계산
        V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);  // 포인트 좌표
        V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);  // 변환 벡터 계산
        V3D P_compensate = imu_state.offset_R_L_I.conjugate() * (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);  // 보정된 포인트 계산 (정확하지 않음!)
        
        // 보정된 포인트 좌표 저장
        it_pcl->x = P_compensate(0);
        it_pcl->y = P_compensate(1);
        it_pcl->z = P_compensate(2);
        if (it_pcl == pcl_2_out.points.begin()) break;  // 첫 번째 포인트까지 처리하면 종료
      }
    }
  }
  pcl_out = pcl_1_out + pcl_2_out;  // 두 LiDAR 포인트 클라우드를 합침
}

void ImuProcess::Process(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr cur_pcl_un_, const bool &multi_lidar, int lidar_num)
{
  if(meas.imu.empty()) {return;};  // IMU 데이터가 비어 있으면 함수 종료
  ROS_ASSERT(meas.lidar != nullptr);  // LiDAR 데이터가 존재하는지 확인

  if (imu_need_init_)
  {
    /// 첫 번째 LiDAR 프레임 초기화
    IMU_init(meas, kf_state, init_iter_num);  // IMU 초기화 함수 호출

    imu_need_init_ = true;  // IMU 초기화가 필요함을 표시
    
    last_imu_   = meas.imu.back();  // 마지막 IMU 데이터를 저장

    if (init_iter_num > MAX_INI_COUNT)  // 초기화 반복 횟수가 최대치를 넘으면
    {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);  // 가속도 공분산을 중력 값에 맞춰 조정
      imu_need_init_ = false;  // IMU 초기화가 더 이상 필요하지 않음

      cov_acc = cov_acc_scale;  // 가속도 공분산 스케일 적용
      cov_gyr = cov_gyr_scale;  // 자이로 공분산 스케일 적용
      ROS_INFO("IMU Initial Done");  // IMU 초기화 완료 메시지 출력
    }
    return;  // 초기화가 완료되었으므로 함수 종료
  }

  if (multi_lidar) UndistortPclMultiLiDAR(meas, kf_state, *cur_pcl_un_);  // 다중 LiDAR일 경우 왜곡 보정 함수 호출
  else UndistortPcl(meas, kf_state, *cur_pcl_un_, lidar_num);  // 단일 LiDAR일 경우 왜곡 보정 함수 호출
  
  // cout<<"[ IMU Process ]: Time: "<<t3 - t1<<endl;  // 처리 시간 디버깅 출력 (주석 처리됨)
}
