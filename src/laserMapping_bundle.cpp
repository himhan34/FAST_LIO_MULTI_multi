// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

/// C++ 헤더
#include <mutex> // 여러 스레드가 자원을 동시에 접근하지 못하도록 상호 배제를 제공
#include <cmath> // 수학적 계산에 필요한 함수 제공
#include <csignal> // 신호 처리를 위한 헤더
#include <unistd.h> // 유닉스 표준 시스템 호출 제공
#include <condition_variable> // 조건 변수로 스레드 간 동기화 제공

/// 모듈 헤더
#include <omp.h> // OpenMP 병렬 프로그래밍 지원

/// Eigen
#include <Eigen/Core> // Eigen 라이브러리의 기본 행렬 및 벡터 연산 지원

/// ROS 헤더
#include <ros/ros.h> // ROS의 기본 기능 제공
#include <geometry_msgs/Vector3.h> // 3차원 벡터 메시지 타입
#include <geometry_msgs/PoseStamped.h> // 포즈 메시지 타입
#include <nav_msgs/Odometry.h> // 로봇의 오도메트리 정보를 위한 메시지 타입
#include <nav_msgs/Path.h> // 로봇의 경로 정보를 위한 메시지 타입
#include <sensor_msgs/PointCloud2.h> // 3D 포인트 클라우드 데이터를 위한 메시지 타입
#include <tf/transform_datatypes.h> // 변환 관련 데이터 타입 및 함수 제공
#include <tf/transform_broadcaster.h> // 좌표 변환 방송을 위한 클래스
#include <livox_ros_driver/CustomMsg.h> // Livox Lidar 사용자 정의 메시지 타입

/// PCL (Point Cloud Library)
#include <pcl/point_cloud.h> // 포인트 클라우드 데이터 구조체 제공
#include <pcl/point_types.h> // 다양한 포인트 타입 제공
#include <pcl/common/transforms.h> // 포인트 클라우드를 변환하는 함수 제공 (transformPointCloud)
#include <pcl/filters/voxel_grid.h> // Voxel Grid 필터 적용을 위한 클래스 제공
#include <pcl_conversions/pcl_conversions.h> // PCL과 ROS 간의 메시지 변환 제공
#include <pcl/io/pcd_io.h> // PCD 파일 입출력을 위한 함수 제공

/// 패키지 내부 헤더
#include "so3_math.h" // SO(3) 수학 연산 관련 함수
#include "IMU_Processing.hpp" // IMU 데이터 처리 관련 클래스 및 함수
#include "preprocess.h" // 데이터 전처리 관련 함수
#include <ikd-Tree/ikd_Tree.h> // ikd-Tree 데이터 구조 및 알고리즘 제공
#include <chrono> // 시간 측정을 위한 chrono 라이브러리
#include <std_msgs/Float32.h> // 32비트 부동 소수점 메시지 타입 제공

using namespace std::chrono; // chrono 네임스페이스를 사용

#define LASER_POINT_COV     (0.001) // 레이저 포인트의 공분산 정의

/**************************/
/// pcd 파일 저장 기능 활성화 여부를 나타내는 변수 (기본값 false)
bool pcd_save_en = false, extrinsic_est_en = true, path_en = true;
/// 이전 결과를 저장하는 배열, 초기값은 0.0
float res_last[100000] = {0.0};
/// 탐지 가능한 범위 설정 (300.0f)
float DET_RANGE = 300.0f;
/// 이동 임계값 설정 (1.5f)
const float MOV_THRESHOLD = 1.5f;

/// LiDAR 및 IMU 주제 이름과 좌표계 프레임을 설정하는 변수
string lid_topic, lid_topic2, imu_topic, map_frame = "map";
/// 다중 LiDAR 사용 여부를 나타내는 변수
bool multi_lidar = false, async_debug = false, publish_tf_results = false;
/// IMU와 LiDAR 간의 외부 매개변수 추정 활성화 여부
bool extrinsic_imu_to_lidars = false;

/// 마지막 LiDAR 결과의 평균 잔차 (residual) 및 총 잔차
double res_mean_last = 0.05, total_residual = 0.0;
/// LiDAR 및 IMU의 마지막 타임스탬프
double last_timestamp_lidar = 0, last_timestamp_lidar2 = 0, last_timestamp_imu = -1.0;
/// 자이로스코프 및 가속도계 공분산 설정
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
/// 표면 필터 크기 설정
double filter_size_surf = 0;
/// 큐브 크기, LiDAR 종료 시간 및 첫 번째 LiDAR 시간 설정
double cube_len = 0, lidar_end_time = 0, lidar_end_time2 = 0, first_lidar_time = 0.0;
/// 효과적인 피처 수와 피처 다운 샘플 크기
int effect_feat_num = 0;
int feats_down_size = 0, NUM_MAX_ITERATIONS = 0, pcd_save_interval = -1, pcd_index = 0;
/// 선택된 표면 포인트 여부를 나타내는 배열, 기본값은 false
bool point_selected_surf[100000] = {0};
/// LiDAR 데이터가 푸시되었는지 여부
bool lidar_pushed = false, flg_exit = false;
/// 스캔, 밀도, 스캔 바디 데이터 발행 여부
bool scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;

/// 삭제해야 하는 박스 포인트의 벡터
vector<BoxPointType> cub_needrm;
/// 가장 가까운 포인트들의 벡터
vector<PointVector>  Nearest_Points;
/// 타임 버퍼 (deque 자료구조)
deque<double>                     time_buffer;
deque<double>                     time_buffer2;
/// LiDAR 데이터 버퍼 (deque 자료구조)
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer2;
/// IMU 데이터 버퍼 (deque 자료구조)
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
/// 버퍼를 보호하기 위한 mutex
mutex mtx_buffer;
/// 버퍼 동기화를 위한 조건 변수
condition_variable sig_buffer;

/// LiDAR 맵에서 추출된 피처 포인트 클라우드를 저장하는 포인터
PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
/// 왜곡을 보정한 피처 포인트 클라우드를 저장하는 포인터
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
/// 다운샘플링된 바디 좌표계에서의 피처 포인트 클라우드를 저장하는 포인터
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
/// 다운샘플링된 월드 좌표계에서의 피처 포인트 클라우드를 저장하는 포인터
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
/// 법선 벡터를 저장하는 포인트 클라우드, 100000 포인트 할당
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
/// 원래의 레이저 클라우드를 저장하는 포인트 클라우드, 100000 포인트 할당
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
/// 상관된 법선 벡터를 저장하는 포인트 클라우드, 100000 포인트 할당
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
/// 저장 대기 중인 포인트 클라우드를 저장하는 포인터
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());

/// 포인트 클라우드의 다운샘플링을 위한 VoxelGrid 필터
pcl::VoxelGrid<PointType> downSizeFilterSurf;
/// KD 트리 데이터 구조
KD_TREE<PointType> ikdtree;
/// 전처리 클래스를 위한 스마트 포인터
shared_ptr<Preprocess> p_pre(new Preprocess());
/// IMU 처리를 위한 스마트 포인터
shared_ptr<ImuProcess> p_imu(new ImuProcess());

/// LiDAR 2와 LiDAR 1 간의 변환 행렬 (항등행렬로 초기화)
Eigen::Matrix4d LiDAR2_wrt_LiDAR1 = Eigen::Matrix4d::Identity();
/// LiDAR 1과 드론 간의 변환 행렬 (항등행렬로 초기화)
Eigen::Matrix4d LiDAR1_wrt_drone = Eigen::Matrix4d::Identity();

/*** EKF (확장 칼만 필터) 입력 및 출력 ***/
/// 측정 데이터를 위한 구조체
MeasureGroup Measures;
/// 확장 칼만 필터 객체 생성 (상태 벡터 크기 12)
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
/// EKF 상태 점
state_ikfom state_point;
/// LiDAR의 위치 벡터
vect3 pos_lid;

/// 경로 메시지를 위한 객체
nav_msgs::Path path;
/// 맵핑 후 오도메트리 데이터를 위한 객체
nav_msgs::Odometry odomAftMapped;
/// 쿼터니언을 위한 메시지 객체
geometry_msgs::Quaternion geoQuat;

/// 로컬 맵 포인트 구조체
BoxPointType LocalMap_Points;
/// 로컬 맵이 초기화되었는지 여부를 나타내는 플래그
bool Localmap_Initialized = false;
/// 첫 번째 LiDAR 스캔 체크 플래그
bool first_lidar_scan_check = false;
/// LiDAR의 평균 스캔 시간
double lidar_mean_scantime = 0.0;
/// 두 번째 LiDAR의 평균 스캔 시간
double lidar_mean_scantime2 = 0.0;
/// LiDAR 스캔 번호
int scan_num = 0;
/// 두 번째 LiDAR 스캔 번호
int scan_num2 = 0;
/// 로컬리저빌리티 벡터 (초기값은 0)
Eigen::Vector3d localizability_vec = Eigen::Vector3d::Zero();

/// 시그널 핸들러 함수, 종료 플래그 설정
void SigHandle(int sig)
{
    flg_exit = true; // 종료 플래그를 true로 설정
    ROS_WARN("catch sig %d", sig); // 시그널을 받아 경고 메시지 출력
    sig_buffer.notify_all(); // 버퍼에 있는 스레드들에게 알림을 전달
}

/// 포인트를 바디 좌표계에서 월드 좌표계로 변환하는 함수
void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    /// 바디 좌표계에서의 포인트 좌표
    V3D p_body(pi->x, pi->y, pi->z);
    /// 변환 행렬을 이용하여 월드 좌표계에서의 포인트 좌표 계산
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    /// 결과를 출력 포인트에 저장
    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity; // 강도 값을 그대로 유지
}

/// 템플릿을 이용한 포인트 변환 함수 (타입을 지원)
template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    /// 바디 좌표계에서의 포인트 좌표
    V3D p_body(pi[0], pi[1], pi[2]);
    /// 변환 행렬을 이용하여 월드 좌표계에서의 포인트 좌표 계산
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    /// 결과를 출력 포인트에 저장
    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

/// RGB 포인트를 바디 좌표계에서 월드 좌표계로 변환하는 함수
void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    /// 바디 좌표계에서의 포인트 좌표
    V3D p_body(pi->x, pi->y, pi->z);
    /// 변환 행렬을 이용하여 월드 좌표계에서의 포인트 좌표 계산
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    /// 결과를 출력 포인트에 저장
    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity; // 강도 값을 그대로 유지
}

/// RGB 포인트를 LiDAR 좌표계에서 IMU 좌표계로 변환하는 함수
void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    /// LiDAR 좌표계에서의 포인트 좌표
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    /// IMU 좌표계로 변환
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    /// 결과를 출력 포인트에 저장
    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity; // 강도 값을 그대로 유지
}

/// LiDAR 맵의 시야 범위를 세분화하는 함수
void lasermap_fov_segment()
{
    /// 삭제해야 할 박스 포인트 벡터를 초기화
    cub_needrm.clear();
    /// LiDAR의 현재 위치를 pos_LiD에 저장
    V3D pos_LiD = pos_lid;

    /// 로컬 맵이 초기화되지 않았다면 초기화
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            /// 로컬 맵의 최소, 최대 좌표를 LiDAR 위치를 기준으로 설정
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        /// 로컬 맵 초기화 완료 표시
        Localmap_Initialized = true;
        return;
    }

    /// 맵의 가장자리까지의 거리를 저장할 배열
    float dist_to_map_edge[3][2];
    /// 맵 이동이 필요한지 여부를 나타내는 플래그
    bool need_move = false;

    /// 각 축에 대해 가장자리와의 거리를 계산하고, 이동이 필요한지 확인
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]); // 최소 좌표에서의 거리
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]); // 최대 좌표에서의 거리
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true; // 이동 필요 여부 확인
    }

    /// 이동이 필요 없으면 함수 종료
    if (!need_move) return;

    /// 새로운 로컬 맵 박스와 임시 박스 포인트 초기화
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;

    /// 이동 거리 계산 (맵이 어느 정도 이동해야 하는지 계산)
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));

    /// 각 축에 대해 로컬 맵을 이동
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            /// 맵의 최소 좌표가 탐지 범위에 너무 가까울 경우, 맵을 이동시키고 삭제할 박스에 추가
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            /// 맵의 최대 좌표가 탐지 범위에 너무 가까울 경우, 맵을 이동시키고 삭제할 박스에 추가
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }

    /// 로컬 맵을 새로운 좌표로 업데이트
    LocalMap_Points = New_LocalMap_Points;

    /// 삭제할 박스 포인트가 있으면 KD 트리에서 해당 박스들을 삭제
    if(cub_needrm.size() > 0) ikdtree.Delete_Point_Boxes(cub_needrm);
}



/// 첫 번째 LiDAR의 포인트 클라우드 콜백 함수
void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    /// 버퍼 보호를 위해 mutex 잠금
    mtx_buffer.lock();
    
    /// 새로운 LiDAR 데이터의 타임스탬프가 마지막 타임스탬프보다 작다면, 루프백이 발생했음을 의미
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer"); // 루프백 발생 시 경고 메시지 출력
        lidar_buffer.clear(); // 버퍼를 비움
    }

    /// 포인트 클라우드 데이터를 저장할 포인터 생성
    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    /// LiDAR 데이터를 전처리하고 결과를 ptr에 저장
    p_pre->process(msg, ptr, 0);
    /// 전처리된 데이터를 LiDAR 버퍼에 추가
    lidar_buffer.push_back(ptr);
    /// 타임스탬프를 시간 버퍼에 추가
    time_buffer.push_back(msg->header.stamp.toSec());
    /// 마지막 LiDAR 타임스탬프를 업데이트
    last_timestamp_lidar = msg->header.stamp.toSec();
    
    /// mutex 잠금 해제
    mtx_buffer.unlock();
    
    /// 조건 변수로 대기 중인 스레드에게 알림을 전달
    sig_buffer.notify_all();
    /// 첫 번째 LiDAR 스캔이 완료되었음을 표시
    first_lidar_scan_check = true;
}

/// 두 번째 LiDAR의 포인트 클라우드 콜백 함수
void standard_pcl_cbk2(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    /// 버퍼 보호를 위해 mutex 잠금
    mtx_buffer.lock();
    
    /// 새로운 LiDAR 데이터의 타임스탬프가 마지막 타임스탬프보다 작다면, 루프백이 발생했음을 의미
    if (msg->header.stamp.toSec() < last_timestamp_lidar2)
    {
        ROS_ERROR("lidar loop back, clear buffer"); // 루프백 발생 시 경고 메시지 출력
        lidar_buffer2.clear(); // 버퍼를 비움
    }

    /// 포인트 클라우드 데이터를 저장할 포인터 생성
    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    /// LiDAR 데이터를 전처리하고 결과를 ptr에 저장
    p_pre->process(msg, ptr, 1);
    /// 전처리된 데이터를 두 번째 LiDAR 버퍼에 추가
    lidar_buffer2.push_back(ptr);
    /// 타임스탬프를 두 번째 시간 버퍼에 추가
    time_buffer2.push_back(msg->header.stamp.toSec());
    /// 마지막 두 번째 LiDAR 타임스탬프를 업데이트
    last_timestamp_lidar2 = msg->header.stamp.toSec();
    
    /// mutex 잠금 해제
    mtx_buffer.unlock();
    
    /// 조건 변수로 대기 중인 스레드에게 알림을 전달
    sig_buffer.notify_all();
    /// 첫 번째 LiDAR 스캔이 완료되었음을 표시
    first_lidar_scan_check = true;
}

/// Livox LiDAR 첫 번째 센서의 포인트 클라우드 콜백 함수
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) 
{
    /// 버퍼 보호를 위해 mutex 잠금
    mtx_buffer.lock();
    
    /// 새로운 LiDAR 데이터의 타임스탬프가 마지막 타임스탬프보다 작으면, 루프백 발생
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer"); // 경고 메시지 출력
        lidar_buffer.clear(); // 버퍼 초기화
    }
    /// LiDAR의 마지막 타임스탬프 업데이트
    last_timestamp_lidar = msg->header.stamp.toSec();

    /// 포인트 클라우드 데이터를 저장할 포인터 생성
    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    /// LiDAR 데이터를 전처리하고 결과를 ptr에 저장
    p_pre->process(msg, ptr, 0);
    /// 전처리된 데이터를 LiDAR 버퍼에 추가
    lidar_buffer.push_back(ptr);
    /// 타임스탬프를 시간 버퍼에 추가
    time_buffer.push_back(last_timestamp_lidar);
    
    /// mutex 잠금 해제
    mtx_buffer.unlock();
    /// 조건 변수로 대기 중인 스레드에게 알림을 전달
    sig_buffer.notify_all();
    /// 첫 번째 LiDAR 스캔 완료 플래그 설정
    first_lidar_scan_check = true;
}

/// Livox LiDAR 두 번째 센서의 포인트 클라우드 콜백 함수
void livox_pcl_cbk2(const livox_ros_driver::CustomMsg::ConstPtr &msg) 
{
    /// 버퍼 보호를 위해 mutex 잠금
    mtx_buffer.lock();
    
    /// 새로운 LiDAR 데이터의 타임스탬프가 마지막 타임스탬프보다 작으면, 루프백 발생
    if (msg->header.stamp.toSec() < last_timestamp_lidar2)
    {
        ROS_ERROR("lidar loop back, clear buffer"); // 경고 메시지 출력
        lidar_buffer2.clear(); // 버퍼 초기화
    }
    /// 두 번째 LiDAR의 마지막 타임스탬프 업데이트
    last_timestamp_lidar2 = msg->header.stamp.toSec();
    
    /// 포인트 클라우드 데이터를 저장할 포인터 생성
    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    /// LiDAR 데이터를 전처리하고 결과를 ptr에 저장
    p_pre->process(msg, ptr, 1);
    /// 전처리된 데이터를 두 번째 LiDAR 버퍼에 추가
    lidar_buffer2.push_back(ptr);
    /// 타임스탬프를 두 번째 시간 버퍼에 추가
    time_buffer2.push_back(last_timestamp_lidar2);
    
    /// mutex 잠금 해제
    mtx_buffer.unlock();
    /// 조건 변수로 대기 중인 스레드에게 알림을 전달
    sig_buffer.notify_all();
    /// 첫 번째 LiDAR 스캔 완료 플래그 설정
    first_lidar_scan_check = true;
}

/// IMU 데이터 콜백 함수
void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    /// 첫 번째 LiDAR 스캔이 완료되지 않으면 반환
    if (!first_lidar_scan_check) return; // 첫 번째 LiDAR 스캔이 완료되지 않았을 경우, IMU 데이터가 너무 많이 축적되는 것을 방지

    /// 새로운 IMU 메시지를 복사
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
    /// IMU 타임스탬프 추출
    double timestamp = msg->header.stamp.toSec();

    /// 버퍼 보호를 위해 mutex 잠금
    mtx_buffer.lock();
    
    /// 새로운 IMU 데이터의 타임스탬프가 마지막 타임스탬프보다 작으면, 루프백 발생
    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer"); // 경고 메시지 출력
        imu_buffer.clear(); // IMU 버퍼 초기화
    }

    /// 마지막 IMU 타임스탬프 업데이트
    last_timestamp_imu = timestamp;

    /// IMU 데이터를 버퍼에 추가
    imu_buffer.push_back(msg);
    
    /// mutex 잠금 해제
    mtx_buffer.unlock();
    /// 조건 변수로 대기 중인 스레드에게 알림을 전달
    sig_buffer.notify_all();
}

/// LiDAR와 IMU 데이터를 동기화하는 함수
bool sync_packages(MeasureGroup &meas)
{
    /// 다중 LiDAR 모드를 사용하는 경우
    if (multi_lidar)
    {
        /// LiDAR 및 IMU 버퍼가 비어 있는 경우 동기화 불가능
        if (lidar_buffer.empty() || lidar_buffer2.empty() || imu_buffer.empty()) {
            return false; // 버퍼가 비어 있으면 false 반환
        }
        /*** LiDAR 스캔 푸시 ***/
        if (!lidar_pushed)
        {
            /// 첫 번째 LiDAR 스캔 데이터를 측정 그룹에 푸시
            meas.lidar = lidar_buffer.front(); // 첫 번째 LiDAR 데이터 할당
            for (size_t i = 1; i < lidar_buffer.size(); i++) // 모든 LiDAR 스캔 병합
            {
                *meas.lidar += *lidar_buffer[i]; // 각 LiDAR 데이터를 합침
            }
            meas.lidar_beg_time = time_buffer.front(); // LiDAR 시작 시간 저장
            /// 포인트 수가 너무 적으면 경고 출력 및 종료 시간 설정
            if (meas.lidar->points.size() <= 1)
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime; // 종료 시간 설정
                ROS_WARN("Too few input point cloud!\n"); // 경고 메시지 출력
            }
            /// 마지막 포인트의 곡률이 평균 스캔 시간보다 작으면 종료 시간 설정
            else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime; // 종료 시간 설정
            }
            /// 그렇지 않으면 스캔 수 증가 및 종료 시간 설정
            else
            {
                scan_num++; // 스캔 수 증가
                lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000); // 종료 시간 계산
                lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num; // 평균 스캔 시간 업데이트
            }
            meas.lidar_end_time = lidar_end_time; // LiDAR 종료 시간 저장

            /// 두 번째 LiDAR 스캔 데이터를 측정 그룹에 푸시
            meas.lidar2 = lidar_buffer2.front(); // 두 번째 LiDAR 데이터 할당
            for (size_t i = 1; i < lidar_buffer2.size(); i++) // 모든 LiDAR 스캔 병합
            {
                *meas.lidar2 += *lidar_buffer2[i]; // 각 LiDAR2 데이터를 합침
            }
            /// LiDAR2 데이터를 LiDAR1 좌표계로 변환
            pcl::transformPointCloud(*meas.lidar2, *meas.lidar2, LiDAR2_wrt_LiDAR1); // 변환 적용
            meas.lidar_beg_time2 = time_buffer2.front(); // LiDAR2 시작 시간 저장
            if (meas.lidar2->points.size() <= 1)
            {
                lidar_end_time2 = meas.lidar_beg_time2 + lidar_mean_scantime2; // 종료 시간 설정
                ROS_WARN("Too few input point cloud!\n"); // 경고 메시지 출력
            }
            else if (meas.lidar2->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime2)
            {
                lidar_end_time2 = meas.lidar_beg_time2 + lidar_mean_scantime2; // 종료 시간 설정
            }
            else
            {
                scan_num2++; // 스캔 수 증가
                lidar_end_time2 = meas.lidar_beg_time2 + meas.lidar2->points.back().curvature / double(1000); // 종료 시간 계산
                lidar_mean_scantime2 += (meas.lidar2->points.back().curvature / double(1000) - lidar_mean_scantime2) / scan_num2; // 평균 스캔 시간 업데이트
            }
            meas.lidar_end_time2 = lidar_end_time2; // LiDAR2 종료 시간 저장

            lidar_pushed = true; // LiDAR 푸시 완료 플래그 설정
        }

        /// IMU 타임스탬프가 LiDAR 종료 시간보다 작으면 동기화 불가능
        if (last_timestamp_imu < lidar_end_time || last_timestamp_imu < lidar_end_time2)
        {
            return false; // IMU 타임스탬프가 종료 시간보다 작으면 false 반환
        }

        /*** IMU 데이터를 푸시하고 버퍼에서 제거 ***/
        double imu_time = imu_buffer.front()->header.stamp.toSec(); // IMU 시간 가져오기
        meas.imu.clear(); // IMU 데이터 초기화
        while ((!imu_buffer.empty()) && (imu_time < lidar_end_time || imu_time < lidar_end_time2))
        {
            imu_time = imu_buffer.front()->header.stamp.toSec(); // IMU 시간 업데이트
            if (imu_time > lidar_end_time && imu_time > lidar_end_time2) break; // 종료 시간 초과 시 종료
            meas.imu.push_back(imu_buffer.front()); // IMU 데이터 추가
            imu_buffer.pop_front(); // 버퍼에서 IMU 데이터 제거
        }

        lidar_buffer.clear(); // LiDAR 버퍼 초기화
        time_buffer.clear(); // LiDAR 시간 버퍼 초기화
        lidar_buffer2.clear(); // 두 번째 LiDAR 버퍼 초기화
        time_buffer2.clear(); // 두 번째 LiDAR 시간 버퍼 초기화

        lidar_pushed = false; // LiDAR 푸시 플래그 초기화
        return true; // 동기화 성공
    }
    /// 단일 LiDAR 모드인 경우
    else
    {
        /// LiDAR 및 IMU 버퍼가 비어 있는 경우 동기화 불가능
        if (lidar_buffer.empty() || imu_buffer.empty()) {
            return false; // 버퍼가 비어 있으면 false 반환
        }
        /*** LiDAR 스캔 푸시 ***/
        if (!lidar_pushed)
        {
            /// 첫 번째 LiDAR 스캔 데이터를 측정 그룹에 푸시
            meas.lidar = lidar_buffer.front(); // LiDAR 데이터 할당
            meas.lidar_beg_time = time_buffer.front(); // LiDAR 시작 시간 저장
            if (meas.lidar->points.size() <= 1)
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime; // 종료 시간 설정
                ROS_WARN("Too few input point cloud!\n"); // 경고 메시지 출력
            }
            else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime; // 종료 시간 설정
            }
            else
            {
                scan_num++; // 스캔 수 증가
                lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000); // 종료 시간 계산
                lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num; // 평균 스캔 시간 업데이트
            }

            meas.lidar_end_time = lidar_end_time; // LiDAR 종료 시간 저장

            lidar_pushed = true; // LiDAR 푸시 완료 플래그 설정
        }

        /// IMU 타임스탬프가 LiDAR 종료 시간보다 작으면 동기화 불가능
        if (last_timestamp_imu < lidar_end_time)
        {
            return false; // IMU 타임스탬프가 종료 시간보다 작으면 false 반환
        }

        /*** IMU 데이터를 푸시하고 버퍼에서 제거 ***/
        double imu_time = imu_buffer.front()->header.stamp.toSec(); // IMU 시간 가져오기
        meas.imu.clear(); // IMU 데이터 초기화
        while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
        {
            imu_time = imu_buffer.front()->header.stamp.toSec(); // IMU 시간 업데이트
            if (imu_time > lidar_end_time) break; // 종료 시간 초과 시 종료
            meas.imu.push_back(imu_buffer.front()); // IMU 데이터 추가
            imu_buffer.pop_front(); // 버퍼에서 IMU 데이터 제거
        }

        lidar_buffer.pop_front(); // LiDAR 데이터 제거
        time_buffer.pop_front(); // 시간 버퍼에서 제거

        lidar_pushed = false; // LiDAR 푸시 플래그 초기화
        return true; // 동기화 성공
    }
}

void map_incremental()
{
    PointVector PointToAdd; // 추가할 포인트를 저장할 벡터
    PointVector PointNoNeedDownsample; // 다운샘플링이 필요 없는 포인트를 저장할 벡터
    PointToAdd.reserve(feats_down_size); // 추가할 포인트 벡터의 용량 예약
    PointNoNeedDownsample.reserve(feats_down_size); // 필요 없는 다운샘플 포인트 벡터의 용량 예약
    for (int i = 0; i < feats_down_size; i++) // 다운샘플된 포인트 수 만큼 반복
    {
        /* 세계 좌표계로 변환 */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i])); // 포인트를 세계 좌표계로 변환
        /* 맵에 추가할 필요가 있는지 결정 */
        if (!Nearest_Points[i].empty()) // 근처 포인트가 비어있지 않으면
        {
            const PointVector &points_near = Nearest_Points[i]; // 근처 포인트 가져오기
            bool need_add = true; // 추가 여부 플래그 초기화
            BoxPointType Box_of_Point; // 포인트의 박스 형태
            PointType downsample_result, mid_point; // 다운샘플 결과와 중간 포인트 초기화
            // 중간 포인트 계산 (그리드 셀 중앙)
            mid_point.x = floor(feats_down_world->points[i].x / filter_size_surf) * filter_size_surf + 0.5 * filter_size_surf;
            mid_point.y = floor(feats_down_world->points[i].y / filter_size_surf) * filter_size_surf + 0.5 * filter_size_surf;
            mid_point.z = floor(feats_down_world->points[i].z / filter_size_surf) * filter_size_surf + 0.5 * filter_size_surf;
            float dist = calc_dist(feats_down_world->points[i], mid_point); // 현재 포인트와 중간 포인트 간 거리 계산
            // 현재 포인트가 중간 포인트와 충분히 멀면 다운샘플링이 필요 없음
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_surf && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_surf && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_surf) {
                PointNoNeedDownsample.push_back(feats_down_world->points[i]); // 다운샘플링 필요 없는 포인트 추가
                continue; // 다음 포인트로 넘어감
            }
            // NUM_MATCH_POINTS 만큼 반복하며 추가 여부 결정
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break; // 근처 포인트가 부족하면 종료
                // 근처 포인트가 중간 포인트보다 가까우면 추가 필요 없음
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false; // 추가하지 않음
                    break; // 루프 종료
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]); // 추가해야 할 경우 포인트 추가
        }
        else // 근처 포인트가 없으면
        {
            PointToAdd.push_back(feats_down_world->points[i]); // 포인트를 추가함
        }
    }
    ikdtree.Add_Points(PointToAdd, true); // 추가할 포인트들을 k-d 트리에 추가
    ikdtree.Add_Points(PointNoNeedDownsample, false); // 필요 없는 다운샘플 포인트를 k-d 트리에 추가
    return; // 함수 종료
}
void publish_frame_world(const ros::Publisher &pubLaserCloudFull, const ros::Publisher &pubLaserCloudFullTransFormed)
{
    if (scan_pub_en) // 스캔 게시가 활성화된 경우
    {
        // dense_pub_en에 따라 적절한 포인트 클라우드 선택
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size(); // 포인트 클라우드의 크기 가져오기
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1)); // 새로운 포인트 클라우드 생성

        for (int i = 0; i < size; i++) // 포인트 클라우드의 각 포인트에 대해
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], // 포인트를 세계 좌표계로 변환
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg; // ROS 메시지 객체 생성
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg); // 포인트 클라우드를 ROS 메시지로 변환
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time); // 타임스탬프 설정
        laserCloudmsg.header.frame_id = map_frame; // 프레임 ID 설정
        pubLaserCloudFull.publish(laserCloudmsg); // 포인트 클라우드 메시지 게시
        
        if (publish_tf_results) // 변환 결과 게시가 활성화된 경우
        {
            PointCloudXYZI::Ptr laserCloudWorldTransFormed(new PointCloudXYZI(size, 1)); // 변환된 포인트 클라우드 생성
            pcl::transformPointCloud(*laserCloudWorld, *laserCloudWorldTransFormed, LiDAR1_wrt_drone); // LiDAR1 기준으로 변환
            sensor_msgs::PointCloud2 laserCloudmsg2; // 또 다른 ROS 메시지 객체 생성
            pcl::toROSMsg(*laserCloudWorldTransFormed, laserCloudmsg2); // 변환된 포인트 클라우드를 ROS 메시지로 변환
            laserCloudmsg2.header.stamp = ros::Time().fromSec(lidar_end_time); // 타임스탬프 설정
            laserCloudmsg2.header.frame_id = map_frame; // 프레임 ID 설정
            pubLaserCloudFullTransFormed.publish(laserCloudmsg2); // 변환된 포인트 클라우드 메시지 게시
        }
    }

    /**************** save map ****************/
    /* 1. 충분한 메모리가 있는지 확인
    /* 2. pcd 저장이 실시간 성능에 영향을 미친다는 점에 유의 **/
    if (pcd_save_en) // PCD 저장이 활성화된 경우
    {
        int size = feats_undistort->points.size(); // 포인트 클라우드 크기 가져오기
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1)); // 새로운 포인트 클라우드 생성

        for (int i = 0; i < size; i++) // 포인트 클라우드의 각 포인트에 대해
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], // 포인트를 세계 좌표계로 변환
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld; // 변환된 포인트 클라우드를 저장할 대기 공간에 추가

        static int scan_wait_num = 0; // 스캔 대기 수 초기화
        scan_wait_num++; // 대기 수 증가
        // PCD 저장 주기 및 대기 수가 기준을 초과한 경우
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
        {
            pcd_index++; // PCD 인덱스 증가
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd")); // PCD 파일 경로 생성
            pcl::PCDWriter pcd_writer; // PCD 작성기 생성
            cout << "current scan saved to /PCD/" << all_points_dir << endl; // 저장된 스캔 출력
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save); // PCD 파일로 포인트 클라우드 저장
            pcl_wait_save->clear(); // 대기 공간 초기화
            scan_wait_num = 0; // 대기 수 초기화
        }
    }
}


void publish_frame_body(const ros::Publisher &pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size(); // 포인트 클라우드의 크기 가져오기
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1)); // IMU 좌표계로 변환된 포인트 클라우드 생성

    for (int i = 0; i < size; i++) // 포인트 클라우드의 각 포인트에 대해
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], // LiDAR 포인트를 IMU 좌표계로 변환
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg; // ROS 메시지 객체 생성
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg); // 포인트 클라우드를 ROS 메시지로 변환
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time); // 타임스탬프 설정
    laserCloudmsg.header.frame_id = "body"; // 프레임 ID를 "body"로 설정
    pubLaserCloudFull_body.publish(laserCloudmsg); // 변환된 포인트 클라우드 메시지 게시
}

void publish_map(const ros::Publisher &pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap; // ROS 메시지 객체 생성
    pcl::toROSMsg(*featsFromMap, laserCloudMap); // 맵 포인트 클라우드를 ROS 메시지로 변환
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time); // 타임스탬프 설정
    laserCloudMap.header.frame_id = map_frame; // 프레임 ID 설정
    pubLaserCloudMap.publish(laserCloudMap); // 맵 포인트 클라우드 메시지 게시
}



template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    return;
}

void publish_visionpose(const ros::Publisher &publisher)
{
    geometry_msgs::PoseStamped msg_out_;
    msg_out_.header.frame_id = map_frame;
    msg_out_.header.stamp = ros::Time().fromSec(lidar_end_time);

    Eigen::Matrix4d current_pose_eig_ = Eigen::Matrix4d::Identity();
    current_pose_eig_.block<3, 3>(0, 0) = state_point.rot.toRotationMatrix();
    current_pose_eig_.block<3, 1>(0, 3) = state_point.pos;
    Eigen::Matrix4d tfed_vision_pose_eig_ = LiDAR1_wrt_drone * current_pose_eig_ * LiDAR1_wrt_drone.inverse(); //note
    msg_out_.pose.position.x = tfed_vision_pose_eig_(0, 3);
    msg_out_.pose.position.y = tfed_vision_pose_eig_(1, 3);
    msg_out_.pose.position.z = tfed_vision_pose_eig_(2, 3);
    Eigen::Quaterniond tfed_quat_(tfed_vision_pose_eig_.block<3, 3>(0, 0));
    msg_out_.pose.orientation.x = tfed_quat_.x();
    msg_out_.pose.orientation.y = tfed_quat_.y();
    msg_out_.pose.orientation.z = tfed_quat_.z();
    msg_out_.pose.orientation.w = tfed_quat_.w();
    publisher.publish(msg_out_);
    return;
}

void publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = map_frame; // 헤더의 프레임 ID 설정
    odomAftMapped.child_frame_id = "body"; // 자식 프레임 ID를 "body"로 설정
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time); // 타임스탬프 설정
    set_posestamp(odomAftMapped.pose); // 포즈 타임스탬프 설정
    pubOdomAftMapped.publish(odomAftMapped); // 오도메트리 메시지 게시
    auto P = kf.get_P(); // 칼만 필터에서 공분산 행렬 P 가져오기
    for (int i = 0; i < 6; i++) // 공분산을 설정하기 위한 반복
    {
        int k = i < 3 ? i + 3 : i - 3; // k 값을 계산
        // 오도메트리 공분산 설정
        odomAftMapped.pose.covariance[i * 6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i * 6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i * 6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i * 6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i * 6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i * 6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br_odom_to_body; // 변환 브로드캐스터 생성
    tf::Transform transform; // 변환 객체 생성
    tf::Quaternion q; // 쿼터니언 객체 생성
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, // 변환 원점 설정
                                    odomAftMapped.pose.pose.position.y,
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w); // 쿼터니언의 w 설정
    q.setX(odomAftMapped.pose.pose.orientation.x); // 쿼터니언의 x 설정
    q.setY(odomAftMapped.pose.pose.orientation.y); // 쿼터니언의 y 설정
    q.setZ(odomAftMapped.pose.pose.orientation.z); // 쿼터니언의 z 설정
    transform.setRotation(q); // 변환의 회전 설정
    br_odom_to_body.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, map_frame, "body")); // 변환 방송
}

void publish_path(const ros::Publisher pubPath)
{
    geometry_msgs::PoseStamped msg_body_pose; // 포즈 스탬프 메시지 객체 생성
    set_posestamp(msg_body_pose); // 포즈 타임스탬프 설정
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time); // 타임스탬프 설정
    msg_body_pose.header.frame_id = map_frame; // 헤더의 프레임 ID 설정

    /*** 경로가 너무 크면 RViz가 중단될 수 있음 ***/
    static int jjj = 0; // 정적 카운터 초기화
    jjj++; // 카운터 증가
    if (jjj % 10 == 0) // 10회마다 경로에 포즈 추가
    {
        path.poses.push_back(msg_body_pose); // 경로에 메시지 추가
        pubPath.publish(path); // 경로 메시지 게시
    }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    laserCloudOri->clear(); // 포인트 클라우드 초기화
    corr_normvect->clear(); // 법선 벡터 클리어
    total_residual = 0.0; // 총 잔차 초기화

    /** 가장 가까운 표면 검색 및 잔차 계산 **/
    #ifdef MP_EN // 멀티 프로세싱이 활성화된 경우
        omp_set_num_threads(MP_PROC_NUM); // 스레드 수 설정
        #pragma omp parallel for // 병렬 루프 시작
    #endif
    for (int i = 0; i < feats_down_size; i++) // 다운샘플된 포인트 수만큼 반복
    {
        PointType &point_body  = feats_down_body->points[i]; // 본체 포인트
        PointType &point_world = feats_down_world->points[i]; // 세계 포인트

        /* 세계 좌표계로 변환 */
        V3D p_body(point_body.x, point_body.y, point_body.z); // 본체 포인트를 V3D로 변환
        V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos); // 글로벌 포인트 계산
        point_world.x = p_global(0); // 변환된 포인트의 x 설정
        point_world.y = p_global(1); // 변환된 포인트의 y 설정
        point_world.z = p_global(2); // 변환된 포인트의 z 설정
        point_world.intensity = point_body.intensity; // 강도 설정

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS); // 거리 저장을 위한 벡터
        auto &points_near = Nearest_Points[i]; // 근처 포인트 참조

        if (ekfom_data.converge) // EKFOM 데이터 수렴 여부 확인
        {
            /** 맵에서 가장 가까운 표면 찾기 **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis); // 최근접 탐색
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true; // 표면 선택 여부 결정
        }

        if (!point_selected_surf[i]) continue; // 선택되지 않은 포인트는 건너뜀

        VF(4) pabcd; // 평면 방정식 계수
        point_selected_surf[i] = false; // 초기화
        if (esti_plane(pabcd, points_near, 0.1f)) // 평면 추정
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3); // 거리 계산
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm()); // 가중치 계산

            if (s > 0.9) // 가중치가 임계값을 초과하면
            {
                point_selected_surf[i] = true; // 표면 선택
                normvec->points[i].x = pabcd(0); // 법선 벡터 x 설정
                normvec->points[i].y = pabcd(1); // 법선 벡터 y 설정
                normvec->points[i].z = pabcd(2); // 법선 벡터 z 설정
                normvec->points[i].intensity = pd2; // 강도 설정
                res_last[i] = abs(pd2); // 잔차 저장
            }
        }
    }
    
    effect_feat_num = 0; // 유효 특징 수 초기화
    localizability_vec = Eigen::Vector3d::Zero(); // 위치 가능성 벡터 초기화
    for (int i = 0; i < feats_down_size; i++) // 모든 포인트에 대해
    {
        if (point_selected_surf[i]) // 선택된 포인트인지 확인
        {
            laserCloudOri->points[effect_feat_num] = feats_down_body->points[i]; // 포인트 추가
            corr_normvect->points[effect_feat_num] = normvec->points[i]; // 법선 벡터 추가
            total_residual += res_last[i]; // 총 잔차 업데이트
            effect_feat_num++; // 유효 특징 수 증가
            localizability_vec += Eigen::Vector3d(normvec->points[i].x, normvec->points[i].y, normvec->points[i].z).array().square().matrix(); // 위치 가능성 벡터 업데이트
        }
    }
    localizability_vec = localizability_vec.cwiseSqrt(); // 제곱근 계산

    if (effect_feat_num < 1) // 유효 포인트가 없으면
    {
        ekfom_data.valid = false; // 유효성 설정
        ROS_WARN("No Effective Points! \n"); // 경고 메시지 출력
        return; // 함수 종료
    }

    res_mean_last = total_residual / effect_feat_num; // 평균 잔차 계산
    
    /*** 측정 자코비안 행렬 H 및 측정 벡터 계산 ***/
    ekfom_data.h_x = MatrixXd::Zero(effect_feat_num, 12); // 자코비안 행렬 초기화
    ekfom_data.h.resize(effect_feat_num); // 측정 벡터 크기 조정

    for (int i = 0; i < effect_feat_num; i++) // 유효 특징 수만큼 반복
    {
        const PointType &laser_p = laserCloudOri->points[i]; // 현재 레이저 포인트
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z); // 포인트를 V3D로 변환
        M3D point_be_crossmat; // 포인트의 교차 행렬
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be); // 교차 행렬 계산
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I; // 변환된 포인트
        M3D point_crossmat; // 변환된 포인트의 교차 행렬
        point_crossmat << SKEW_SYM_MATRX(point_this); // 교차 행렬 계산

        /*** 가장 가까운 표면/코너의 법선 벡터 가져오기 ***/
        const PointType &norm_p = corr_normvect->points[i]; // 법선 포인트
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z); // 법선 벡터

        /*** 측정 자코비안 행렬 H 계산 ***/
        V3D C(s.rot.conjugate() * norm_vec); // 법선 벡터의 변환
        V3D A(point_crossmat * C); // 자코비안 A 계산
        if (extrinsic_est_en) // 외부 추정이 활성화된 경우
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); // 자코비안 B 계산
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C); // 자코비안 행렬 설정
        }
        else // 외부 추정이 비활성화된 경우
        {
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0; // 자코비안 행렬 설정
        }

        /*** 측정: 가장 가까운 표면/코너까지의 거리 ***/
        ekfom_data.h(i) = -norm_p.intensity; // 측정 벡터 설정
    }
}



int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping"); // ROS 노드 초기화 및 이름 설정
    ros::NodeHandle nh; // 노드 핸들 생성

    // 임시 변수 선언
    vector<double> extrinT(3, 0.0); // 외부 변환 T 초기화
    vector<double> extrinR(9, 0.0); // 외부 변환 R 초기화
    vector<double> extrinT2(3, 0.0); // 두 번째 외부 변환 T 초기화
    vector<double> extrinR2(9, 0.0); // 두 번째 외부 변환 R 초기화
    vector<double> extrinT3(3, 0.0); // 세 번째 외부 변환 T 초기화
    vector<double> extrinR3(9, 0.0); // 세 번째 외부 변환 R 초기화
    vector<double> extrinT4(3, 0.0); // 네 번째 외부 변환 T 초기화
    vector<double> extrinR4(9, 0.0); // 네 번째 외부 변환 R 초기화

    nh.param<int>("common/max_iteration", NUM_MAX_ITERATIONS, 4); // 최대 반복 횟수 파라미터 가져오기
    nh.param<bool>("common/async_debug", async_debug, false); // 비동기 디버깅 활성화 여부 가져오기
    nh.param<bool>("common/multi_lidar", multi_lidar, true); // 다중 LiDAR 활성화 여부 가져오기
    nh.param<bool>("common/publish_tf_results", publish_tf_results, true); // TF 결과 게시 여부 가져오기
    nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar"); // LiDAR 토픽 이름 가져오기
    nh.param<string>("common/lid_topic2", lid_topic2, "/livox/lidar"); // 두 번째 LiDAR 토픽 이름 가져오기
    nh.param<string>("common/imu_topic", imu_topic, "/livox/imu"); // IMU 토픽 이름 가져오기
    nh.param<string>("common/map_frame", map_frame, "map"); // 맵 프레임 이름 가져오기

    nh.param<double>("preprocess/filter_size_surf", filter_size_surf, 0.5); // 필터 사이즈 파라미터 가져오기
    nh.param<int>("preprocess/point_filter_num", p_pre->point_filter_num[0], 2); // 포인트 필터 수 가져오기
    nh.param<int>("preprocess/point_filter_num2", p_pre->point_filter_num[1], 2); // 두 번째 포인트 필터 수 가져오기
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type[0], AVIA); // LiDAR 타입 가져오기
    nh.param<int>("preprocess/lidar_type2", p_pre->lidar_type[1], AVIA); // 두 번째 LiDAR 타입 가져오기
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS[0], 16); // 스캔 라인 수 가져오기
    nh.param<int>("preprocess/scan_line2", p_pre->N_SCANS[1], 16); // 두 번째 스캔 라인 수 가져오기
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE[0], 10); // 스캔 속도 가져오기
    nh.param<int>("preprocess/scan_rate2", p_pre->SCAN_RATE[1], 10); // 두 번째 스캔 속도 가져오기
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit[0], US); // 타임스탬프 단위 가져오기
    nh.param<int>("preprocess/timestamp_unit2", p_pre->time_unit[1], US); // 두 번째 타임스탬프 단위 가져오기
    nh.param<double>("preprocess/blind", p_pre->blind[0], 0.01); // 블라인드 값 가져오기
    nh.param<double>("preprocess/blind2", p_pre->blind[1], 0.01); // 두 번째 블라인드 값 가져오기
    p_pre->set(); // 전처리 설정

    nh.param<double>("mapping/cube_side_length", cube_len, 200.0); // 맵 큐브의 한 변 길이 가져오기
    nh.param<float>("mapping/det_range", DET_RANGE, 300.f); // 감지 범위 가져오기
    nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1); // 자이로 공분산 가져오기
    nh.param<double>("mapping/acc_cov", acc_cov, 0.1); // 가속도 공분산 가져오기
    nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001); // 자이로 바이어스 공분산 가져오기
    nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001); // 가속도 바이어스 공분산 가져오기
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true); // 외부 추정 활성화 여부 가져오기
    nh.param<bool>("mapping/extrinsic_imu_to_lidars", extrinsic_imu_to_lidars, true); // IMU를 LiDAR로 변환 여부 가져오기
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>()); // 외부 변환 T 가져오기
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>()); // 외부 변환 R 가져오기
    nh.param<vector<double>>("mapping/extrinsic_T2", extrinT2, vector<double>()); // 두 번째 외부 변환 T 가져오기
    nh.param<vector<double>>("mapping/extrinsic_R2", extrinR2, vector<double>()); // 두 번째 외부 변환 R 가져오기
    nh.param<vector<double>>("mapping/extrinsic_T_L2_wrt_L1", extrinT3, vector<double>()); // LiDAR2가 LiDAR1에 대한 외부 변환 T 가져오기
    nh.param<vector<double>>("mapping/extrinsic_R_L2_wrt_L1", extrinR3, vector<double>()); // LiDAR2가 LiDAR1에 대한 외부 변환 R 가져오기
    nh.param<vector<double>>("mapping/extrinsic_T_L1_wrt_drone", extrinT4, vector<double>()); // LiDAR1이 드론에 대한 외부 변환 T 가져오기
    nh.param<vector<double>>("mapping/extrinsic_R_L1_wrt_drone", extrinR4, vector<double>()); // LiDAR1이 드론에 대한 외부 변환 R 가져오기

    nh.param<bool>("publish/path_en", path_en, true); // 경로 게시 활성화 여부 가져오기
    nh.param<bool>("publish/scan_publish_en", scan_pub_en, true); // 스캔 게시 활성화 여부 가져오기
    nh.param<bool>("publish/dense_publish_en", dense_pub_en, true); // 밀집 게시 활성화 여부 가져오기
    nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en, true); // 본체 프레임 게시 활성화 여부 가져오기
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false); // PCD 저장 활성화 여부 가져오기
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1); // PCD 저장 간격 가져오기
    
    /*** 변수 정의 ***/
    path.header.stamp = ros::Time::now(); // 경로 타임스탬프 설정
    path.header.frame_id = map_frame; // 경로 프레임 ID 설정
    memset(point_selected_surf, true, sizeof(point_selected_surf)); // 선택된 포인트 초기화
    memset(res_last, -1000.0f, sizeof(res_last)); // 마지막 잔차 초기화
    downSizeFilterSurf.setLeafSize(filter_size_surf, filter_size_surf, filter_size_surf); // 다운샘플링 필터 설정
    ikdtree.set_downsample_param(filter_size_surf); // k-d 트리의 다운샘플링 파라미터 설정

    V3D Lidar_T_wrt_IMU(Zero3d); // LiDAR의 IMU에 대한 변환 초기화
    M3D Lidar_R_wrt_IMU(Eye3d); // LiDAR의 IMU에 대한 회전 초기화
    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT); // 변환 T 설정
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR); // 변환 R 설정
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU); // IMU에 외부 변환 설정
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov)); // 자이로 공분산 설정
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov)); // 가속도 공분산 설정
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov)); // 자이로 바이어스 공분산 설정
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov)); // 가속도 바이어스 공분산 설정


    // 다중 LiDAR에 대한 변환을 위한 코드
    if (multi_lidar)
    {
        if (extrinsic_imu_to_lidars) // IMU에서 LiDAR로 변환이 활성화된 경우
        {
            Eigen::Matrix4d Lidar_wrt_IMU = Eigen::Matrix4d::Identity(); // LiDAR의 IMU에 대한 변환 행렬 초기화
            Eigen::Matrix4d Lidar2_wrt_IMU = Eigen::Matrix4d::Identity(); // 두 번째 LiDAR의 IMU에 대한 변환 행렬 초기화
            V3D LiDAR2_T_wrt_IMU; LiDAR2_T_wrt_IMU << VEC_FROM_ARRAY(extrinT2); // LiDAR2의 IMU에 대한 위치 벡터 설정
            M3D LiDAR2_R_wrt_IMU; LiDAR2_R_wrt_IMU << MAT_FROM_ARRAY(extrinR2); // LiDAR2의 IMU에 대한 회전 행렬 설정
            Lidar_wrt_IMU.block<3, 3>(0, 0) = Lidar_R_wrt_IMU; // LiDAR의 회전 행렬 설정
            Lidar_wrt_IMU.block<3, 1>(0, 3) = Lidar_T_wrt_IMU; // LiDAR의 위치 벡터 설정
            Lidar2_wrt_IMU.block<3, 3>(0, 0) = LiDAR2_R_wrt_IMU; // LiDAR2의 회전 행렬 설정
            Lidar2_wrt_IMU.block<3, 1>(0, 3) = LiDAR2_T_wrt_IMU; // LiDAR2의 위치 벡터 설정
            LiDAR2_wrt_LiDAR1 = Lidar_wrt_IMU.inverse() * Lidar2_wrt_IMU; // LiDAR2를 LiDAR1에 대한 변환 계산
        }
        else // IMU에서 LiDAR로 변환이 비활성화된 경우
        {
            V3D LiDAR2_T_wrt_LiDAR1; LiDAR2_T_wrt_LiDAR1 << VEC_FROM_ARRAY(extrinT3); // LiDAR2가 LiDAR1에 대한 위치 벡터 설정
            M3D Lidar2_R_wrt_LiDAR1; Lidar2_R_wrt_LiDAR1 << MAT_FROM_ARRAY(extrinR3); // LiDAR2가 LiDAR1에 대한 회전 행렬 설정
            LiDAR2_wrt_LiDAR1.block<3, 3>(0, 0) = Lidar2_R_wrt_LiDAR1; // LiDAR2의 회전 행렬 설정
            LiDAR2_wrt_LiDAR1.block<3, 1>(0, 3) = LiDAR2_T_wrt_LiDAR1; // LiDAR2의 위치 벡터 설정
        }
        cout << "\033[32;1mMulti LiDAR on!" << endl; // 다중 LiDAR 활성화 메시지 출력
        cout << "lidar_type[0]: " << p_pre->lidar_type[0] << ", " << "lidar_type[1]: " << p_pre->lidar_type[1] << endl << endl; // LiDAR 타입 출력
        cout << "L2 wrt L1 TF: " << endl << LiDAR2_wrt_LiDAR1 << "\033[0m" << endl << endl; // LiDAR2가 LiDAR1에 대한 변환 출력
    }
    if (publish_tf_results) // TF 결과 게시가 활성화된 경우
    {
        V3D LiDAR1_T_wrt_drone; LiDAR1_T_wrt_drone << VEC_FROM_ARRAY(extrinT4); // LiDAR1이 드론에 대한 위치 벡터 설정
        M3D LiDAR2_R_wrt_drone; LiDAR2_R_wrt_drone << MAT_FROM_ARRAY(extrinR4); // LiDAR1이 드론에 대한 회전 행렬 설정
        LiDAR1_wrt_drone.block<3, 3>(0, 0) = LiDAR2_R_wrt_drone; // LiDAR1의 회전 행렬 설정
        LiDAR1_wrt_drone.block<3, 1>(0, 3) = LiDAR1_T_wrt_drone; // LiDAR1의 위치 벡터 설정
        cout << "\033[32;1mLiDAR wrt Drone:" << endl; // LiDAR와 드론 간 변환 출력
        cout << LiDAR1_wrt_drone << "\033[0m" << endl << endl; // LiDAR와 드론 간 변환 행렬 출력
    }
    
    double epsi[23] = {0.001}; // 공차 배열 초기화
    fill(epsi, epsi + 23, 0.001); // 공차 배열 값 설정
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi); // 칼만 필터 초기화
    
    /*** ROS 구독 초기화 ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type[0] == AVIA ? \
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 200000, standard_pcl_cbk); // LiDAR 타입에 따라 콜백 함수 설정
    ros::Subscriber sub_pcl2; // 두 번째 LiDAR 구독자 선언
    if (multi_lidar) // 다중 LiDAR인 경우
    {
        sub_pcl2 = p_pre->lidar_type[1] == AVIA ? \
            nh.subscribe(lid_topic2, 200000, livox_pcl_cbk2) : \
            nh.subscribe(lid_topic2, 200000, standard_pcl_cbk2); // 두 번째 LiDAR에 대한 콜백 함수 설정
    }
    
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk); // IMU 데이터 구독자 설정
    
    // 포인트 클라우드를 게시하기 위한 퍼블리셔 설정
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>
                ("/cloud_registered", 100000); // 등록된 포인트 클라우드 퍼블리셔
    ros::Publisher pubLaserCloudFullTransformed = nh.advertise<sensor_msgs::PointCloud2>
                ("/cloud_registered_tf", 100000); // 변환된 포인트 클라우드 퍼블리셔
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>
                ("/cloud_registered_body", 100000); // 본체 좌표계의 등록된 포인트 클라우드 퍼블리셔
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
                ("/Laser_map", 100000); // 라이다 맵 퍼블리셔
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> 
                ("/Odometry", 100000); // 매핑 후 오도메트리 퍼블리셔
    ros::Publisher pubMavrosVisionPose = nh.advertise<geometry_msgs::PoseStamped> 
                ("/mavros/vision_pose/pose", 100000); // MAVROS 비전 포즈 퍼블리셔
    ros::Publisher pubPath = nh.advertise<nav_msgs::Path> 
                ("/path", 100000); // 경로 퍼블리셔
    ros::Publisher pubCaclTime = nh.advertise<std_msgs::Float32> 
                ("/calc_time", 100000); // 계산 시간 퍼블리셔
    ros::Publisher pubPointNum = nh.advertise<std_msgs::Float32> 
                ("/point_number", 100000); // 포인트 수 퍼블리셔
    ros::Publisher pubLocalizabilityX = nh.advertise<std_msgs::Float32> 
                ("/localizability_x", 100000); // 위치 가능성 X 퍼블리셔
    ros::Publisher pubLocalizabilityY = nh.advertise<std_msgs::Float32> 
                ("/localizability_y", 100000); // 위치 가능성 Y 퍼블리셔
    ros::Publisher pubLocalizabilityZ = nh.advertise<std_msgs::Float32> 
                ("/localizability_z", 100000); // 위치 가능성 Z 퍼블리셔
    
    //------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle); // SIGINT 신호에 대한 핸들러 설정
    ros::Rate rate(5000); // 주기를 5000Hz로 설정
    bool status = ros::ok(); // ROS 노드가 정상인지 확인
    
      while (status) // ROS가 정상인 동안 반복
    {
        if (flg_exit) break; // 종료 플래그가 설정된 경우 루프 종료
        ros::spinOnce(); // 콜백 함수 실행
    
        if(sync_packages(Measures)) // 패키지가 동기화되면
        {
            high_resolution_clock::time_point t1 = high_resolution_clock::now(); // 현재 시간 기록
            p_imu->Process(Measures, kf, feats_undistort, multi_lidar); // IMU 데이터 처리
            state_point = kf.get_x(); // 칼만 필터 상태 추정
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I; // LiDAR 위치 계산
    
            if (feats_undistort->empty() || (feats_undistort == NULL)) // 다운샘플링된 포인트가 비어있으면
            {
                ROS_WARN("No point, skip this scan!\n"); // 경고 메시지 출력
                continue; // 다음 반복으로 넘어감
            }
    
            /*** LiDAR FOV에서 맵 세그먼트화 ***/
            lasermap_fov_segment(); // LiDAR 시야 내에서 맵 세그먼트화 수행
    
            /*** 스캔에서 피처 포인트 다운샘플링 ***/
            downSizeFilterSurf.setInputCloud(feats_undistort); // 입력 포인트 클라우드 설정
            downSizeFilterSurf.filter(*feats_down_body); // 다운샘플링 수행
            feats_down_size = feats_down_body->points.size(); // 다운샘플링된 포인트 수 저장
    
            /*** 맵 kdtree 초기화 ***/
            if (ikdtree.Root_Node == nullptr) // kdtree의 루트 노드가 비어있으면
            {
                if (feats_down_size > 5) // 포인트 수가 5보다 많으면
                {
                    feats_down_world->resize(feats_down_size); // 세계 포인트 클라우드 크기 조정
                    for (int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i])); // 포인트를 세계 좌표계로 변환
                    }                    
                    ikdtree.Add_Points(feats_down_world->points, true); // 포인트를 kdtree에 추가
                }
                continue; // 다음 반복으로 넘어감
            }
    
            /*** ICP 및 반복 칼만 필터 업데이트 ***/
            if (feats_down_size < 5) // 다운샘플링된 포인트 수가 5보다 적으면
            {
                ROS_WARN("No point, skip this scan!\n"); // 경고 메시지 출력
                continue; // 다음 반복으로 넘어감
            }
            
            normvec->resize(feats_down_size); // 법선 벡터 크기 조정
            feats_down_world->resize(feats_down_size); // 세계 포인트 클라우드 크기 조정
            Nearest_Points.resize(feats_down_size); // 근처 포인트 벡터 크기 조정
            /*** 반복 상태 추정 ***/
            double solve_H_time = 0; // 측정 자코비안 계산 시간을 위한 변수 초기화
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time); // 동적 상태 업데이트
            state_point = kf.get_x(); // 칼만 필터 상태 추정
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I; // LiDAR 위치 계산
            geoQuat.x = state_point.rot.coeffs()[0]; // 쿼터니언 x 설정
            geoQuat.y = state_point.rot.coeffs()[1]; // 쿼터니언 y 설정
            geoQuat.z = state_point.rot.coeffs()[2]; // 쿼터니언 z 설정
            geoQuat.w = state_point.rot.coeffs()[3]; // 쿼터니언 w 설정
    
            /******* 오도메트리 게시 *******/
            if (publish_tf_results) publish_visionpose(pubMavrosVisionPose); // 비전 포즈 게시
            publish_odometry(pubOdomAftMapped); // 오도메트리 게시
    
            /*** 피처 포인트를 맵 kdtree에 추가 ***/
            map_incremental(); // 맵 업데이트
    
            if (0) // 맵 포인트를 보고 싶으면 "if(1)"로 변경
            {
                PointVector().swap(ikdtree.PCL_Storage); // kdtree 저장 공간 초기화
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD); // kdtree 평탄화
                featsFromMap->clear(); // 맵 포인트 클라우드 초기화
                featsFromMap->points = ikdtree.PCL_Storage; // kdtree의 포인트 저장
            }
            /******* 포인트 게시 *******/
            if (path_en) publish_path(pubPath); // 경로 게시
            if (scan_pub_en || pcd_save_en) publish_frame_world(pubLaserCloudFull, pubLaserCloudFullTransformed); // 포인트 클라우드 게시
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body); // 본체 프레임 포인트 클라우드 게시
            // publish_map(pubLaserCloudMap); // 맵 게시 (주석 처리됨)
            high_resolution_clock::time_point t2 = high_resolution_clock::now(); // 종료 시간 기록
            auto duration = duration_cast<microseconds>(t2 - t1).count() / 1000.0; // 실행 시간 계산
            std_msgs::Float32 calc_time; // 계산 시간 메시지 객체 생성
            calc_time.data = duration; // 계산 시간 설정
            pubCaclTime.publish(calc_time); // 계산 시간 게시
            std_msgs::Float32 point_num; // 포인트 수 메시지 객체 생성
            point_num.data = feats_down_size; // 포인트 수 설정
            pubPointNum.publish(point_num); // 포인트 수 게시
            std_msgs::Float32 localizability_x, localizability_y, localizability_z; // 위치 가능성 메시지 객체 생성
            localizability_x.data = localizability_vec(0); // X 위치 가능성 설정
            localizability_y.data = localizability_vec(1); // Y 위치 가능성 설정
            localizability_z.data = localizability_vec(2); // Z 위치 가능성 설정
            pubLocalizabilityX.publish(localizability_x); // X 위치 가능성 게시
            pubLocalizabilityY.publish(localizability_y); // Y 위치 가능성 게시
            pubLocalizabilityZ.publish(localizability_z); // Z 위치 가능성 게시
        }
        status = ros::ok(); // ROS 상태 확인
        rate.sleep(); // 주기적 대기
    }

  /**************** 맵 저장 ****************/
/* 1. 충분한 메모리가 있는지 확인
/* 2. PCD 저장이 실시간 성능에 크게 영향을 미침 **/
if (pcl_wait_save->size() > 0 && pcd_save_en) // PCD 저장이 활성화되고 대기 중인 PCD가 있는 경우
{
    string file_name = string("scans.pcd"); // 저장할 PCD 파일 이름 설정
    string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name); // 전체 PCD 파일 경로 설정
    pcl::PCDWriter pcd_writer; // PCD 작성기 객체 생성
    cout << "current scan saved to /PCD/" << file_name << endl; // 저장된 PCD 파일 경로 출력
    pcd_writer.writeBinary(all_points_dir, *pcl_wait_save); // PCD 파일로 포인트 클라우드 저장
}

return 0; // 프로그램 종료
}
