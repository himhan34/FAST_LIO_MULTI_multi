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

/// C++ 기본 헤더
#include <mutex>                 // 멀티스레딩을 위한 mutex
#include <cmath>                 // 수학 관련 함수
#include <csignal>               // 신호 처리 관련
#include <unistd.h>              // 유닉스 시스템 콜
#include <condition_variable>    // 조건 변수 사용을 위한 헤더
/// 모듈 헤더
#include <omp.h>                 // OpenMP 병렬 프로그래밍 지원
/// Eigen
#include <Eigen/Core>            // Eigen 라이브러리, 선형대수 계산
/// ROS
#include <ros/ros.h>             // ROS 노드 관련 기본 헤더
#include <geometry_msgs/Vector3.h> // 벡터3 메시지 타입
#include <geometry_msgs/PoseStamped.h> // PoseStamped 메시지 타입
#include <nav_msgs/Odometry.h>     // 오도메트리 메시지 타입
#include <nav_msgs/Path.h>         // 경로 메시지 타입
#include <sensor_msgs/PointCloud2.h> // 포인트 클라우드 메시지 타입
#include <tf/transform_datatypes.h> // 좌표 변환 데이터 타입
#include <tf/transform_broadcaster.h> // 좌표 변환 브로드캐스트
#include <livox_ros_driver/CustomMsg.h> // Livox LiDAR 드라이버의 커스텀 메시지
/// PCL (Point Cloud Library)
#include <pcl/point_cloud.h>      // 포인트 클라우드 관련 헤더
#include <pcl/point_types.h>      // 포인트 타입 정의
#include <pcl/common/transforms.h> // 포인트 클라우드 변환
#include <pcl/filters/voxel_grid.h> // Voxel 그리드 필터 사용
#include <pcl_conversions/pcl_conversions.h> // PCL 포인트 클라우드와 ROS 메시지 간 변환
#include <pcl/io/pcd_io.h>        // PCD 파일 입출력
/// 패키지 전용 헤더
#include "so3_math.h"            // SO3 수학 함수 헤더
#include "IMU_Processing.hpp"    // IMU 처리 관련 헤더
#include "preprocess.h"          // 전처리 관련 헤더
#include <ikd-Tree/ikd_Tree.h>   // KD-트리 라이브러리

#include <chrono>                // 시간 관련 함수 및 클래스
#include <std_msgs/Float32.h>    // 표준 Float32 메시지 타입
using namespace std::chrono;     // chrono 네임스페이스 사용

#define LASER_POINT_COV     (0.001) // 레이저 포인트의 공분산 값 상수 정의


/**************************/  
// 변수 선언 및 초기화
bool pcd_save_en = false, extrinsic_est_en = true, path_en = true; // PCD 저장 여부, 외부 요소 추정 활성화 여부, 경로 활성화 여부
float res_last[100000] = {0.0}; // 마지막 잔차 값 저장
float DET_RANGE = 300.0f; // 탐지 범위 설정
const float MOV_THRESHOLD = 1.5f; // 움직임 임계값

// Lidar 및 IMU 토픽, 맵 프레임 설정
string lid_topic, lid_topic2, imu_topic, map_frame = "map"; 
bool multi_lidar = false, async_debug = false, publish_tf_results = false; // 다중 라이다 사용 여부, 비동기 디버그 모드, TF 결과 발행 여부
bool extrinsic_imu_to_lidars = false; // IMU와 라이다 간 외부 요소 적용 여부

// 잔차 및 타임스탬프 변수 초기화
double res_mean_last = 0.05, total_residual = 0.0; 
double last_timestamp_lidar = 0, last_timestamp_lidar2 = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001; // 자이로스코프 및 가속도계 공분산 설정
double filter_size_surf = 0; // 필터 크기 설정
double cube_len = 0, lidar_end_time = 0, first_lidar_time = 0.0; // 맵 큐브 길이, 라이다 종료 시간, 첫 라이다 시간
int    effect_feat_num = 0; // 유효 특징점 수
int    feats_down_size = 0, NUM_MAX_ITERATIONS = 0, pcd_save_interval = -1, pcd_index = 0; // 다운샘플링 후 특징점 수, 최대 반복 횟수, PCD 저장 간격 및 인덱스
bool   point_selected_surf[100000] = {0}; // 표면에서 선택된 포인트 여부
bool   lidar_pushed = false, flg_exit = false; // 라이다 데이터가 푸시되었는지 여부, 종료 플래그
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false; // 스캔 발행 여부, 밀도 스캔 발행 여부, 바디 프레임 스캔 발행 여부

// 필요 제거 큐브 및 최근접 점들
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 

// 버퍼 및 관련 변수들
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer; // 라이다 포인트 클라우드 버퍼
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;   // IMU 데이터 버퍼
mutex mtx_buffer; // 버퍼를 위한 뮤텍스
condition_variable sig_buffer; // 조건 변수

// 포인트 클라우드 관련 변수들
PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());   // 맵에서 가져온 특징점들
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI()); // 왜곡을 보정한 특징점들
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI()); // 다운샘플링 후 바디 프레임에서의 특징점들
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI()); // 다운샘플링 후 월드 프레임에서의 특징점들
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1)); // 표면 법선 벡터들
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1)); // 원본 레이저 클라우드
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1)); // 보정된 법선 벡터들
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI()); // 저장을 위한 포인트 클라우드

// Voxel 그리드 필터 및 KD-트리 설정
pcl::VoxelGrid<PointType> downSizeFilterSurf;
KD_TREE<PointType> ikdtree;

// 전처리 및 IMU 처리 객체
shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

// 라이다 간 및 라이다와 드론 간 변환 행렬
Eigen::Matrix4d LiDAR2_wrt_LiDAR1 = Eigen::Matrix4d::Identity(); // 라이다2가 라이다1에 대해 어떻게 배치되었는지
Eigen::Matrix4d LiDAR1_wrt_drone = Eigen::Matrix4d::Identity();  // 라이다1이 드론에 대해 어떻게 배치되었는지


/*** EKF 입력 및 출력 ***/
MeasureGroup Measures; // 측정 데이터 그룹
esekfom::esekf<state_ikfom, 12, input_ikfom> kf; // 확장 칼만 필터 객체
state_ikfom state_point; // 필터에서 출력되는 상태 정보
vect3 pos_lid; // 라이다의 위치

nav_msgs::Path path; // 경로 메시지
nav_msgs::Odometry odomAftMapped; // 맵핑 후의 오도메트리
geometry_msgs::Quaternion geoQuat; // 지오메트리 쿼터니언(회전)

BoxPointType LocalMap_Points; // 로컬 맵 포인트 타입
bool Localmap_Initialized = false; // 로컬 맵 초기화 여부
bool first_lidar_scan_check = false; // 첫 라이다 스캔 여부 확인
bool lidar1_ikd_init = false, lidar2_ikd_init = false; // 라이다 1, 2의 KD-트리 초기화 여부
int current_lidar_num = 1; // 현재 사용하는 라이다 번호
double lidar_mean_scantime = 0.0; // 라이다 평균 스캔 시간
double lidar_mean_scantime2 = 0.0; // 라이다 2 평균 스캔 시간
int scan_num = 0; // 스캔 횟수
int scan_num2 = 0; // 라이다 2의 스캔 횟수
Eigen::Vector3d localizability_vec = Eigen::Vector3d::Zero(); // 로컬라이저빌리티 벡터

// 시그널 핸들러 함수 (시스템 종료 시 동작)
void SigHandle(int sig)
{
    flg_exit = true; // 종료 플래그 설정
    ROS_WARN("catch sig %d", sig); // 잡힌 시그널 출력
    sig_buffer.notify_all(); // 모든 대기중인 스레드에 알림
}

// 로봇 바디 프레임 좌표를 월드 프레임 좌표로 변환하는 함수
void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z); // 입력 포인트의 바디 좌표
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos); // 바디 좌표를 월드 좌표로 변환

    po->x = p_global(0); // 변환된 x 좌표
    po->y = p_global(1); // 변환된 y 좌표
    po->z = p_global(2); // 변환된 z 좌표
    po->intensity = pi->intensity; // 포인트의 강도 값 그대로 복사
}


// 좌표를 바디 프레임에서 월드 프레임으로 변환하는 템플릿 함수
template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]); // 입력 포인트를 바디 프레임 좌표로 변환
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos); // 바디 프레임 좌표를 월드 프레임으로 변환

    po[0] = p_global(0); // 변환된 x 좌표
    po[1] = p_global(1); // 변환된 y 좌표
    po[2] = p_global(2); // 변환된 z 좌표
}

// RGB 포인트를 바디 프레임에서 월드 프레임으로 변환하는 함수
void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z); // 입력 포인트의 바디 프레임 좌표
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos); // 바디 프레임 좌표를 월드 프레임으로 변환

    po->x = p_global(0); // 변환된 x 좌표
    po->y = p_global(1); // 변환된 y 좌표
    po->z = p_global(2); // 변환된 z 좌표
    po->intensity = pi->intensity; // 입력 포인트의 강도 값을 그대로 복사
}

// RGB 포인트를 라이다 프레임에서 IMU 프레임으로 변환하는 함수
void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z); // 입력 포인트의 라이다 프레임 좌표
    V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I); // 라이다 프레임 좌표를 IMU 프레임으로 변환

    po->x = p_body_imu(0); // 변환된 x 좌표
    po->y = p_body_imu(1); // 변환된 y 좌표
    po->z = p_body_imu(2); // 변환된 z 좌표
    po->intensity = pi->intensity; // 입력 포인트의 강도 값을 그대로 복사
}

// 라이다 FOV 내에서 맵을 세그먼트로 나누는 함수
void lasermap_fov_segment()
{
    cub_needrm.clear(); // 제거할 큐브 초기화
    V3D pos_LiD = pos_lid; // 라이다의 현재 위치
    if (!Localmap_Initialized) {
        for (int i = 0; i < 3; i++) {
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0; // 맵의 최소 경계 설정
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0; // 맵의 최대 경계 설정
        }
        Localmap_Initialized = true; // 로컬 맵이 초기화되었음을 표시
        return;
    }

    float dist_to_map_edge[3][2]; // 각 축의 경계까지 거리 계산
    bool need_move = false; // 맵 이동이 필요한지 여부

    // 라이다 위치와 맵 경계 간의 거리를 계산하여 이동 필요 여부 확인
    for (int i = 0; i < 3; i++) {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]); // 최소 경계까지 거리
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]); // 최대 경계까지 거리
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
            need_move = true; // 맵 이동이 필요하다고 표시
    }

    if (!need_move) return; // 맵 이동이 필요 없으면 함수 종료

    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD - 1)));

    for (int i = 0; i < 3; i++) {
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE) {
            New_LocalMap_Points.vertex_max[i] -= mov_dist; // 맵을 이동
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints); // 제거할 큐브 추가
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) {
            New_LocalMap_Points.vertex_max[i] += mov_dist; // 맵을 이동
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints); // 제거할 큐브 추가
        }
    }

    LocalMap_Points = New_LocalMap_Points; // 맵의 새로운 경계 설정

    if (cub_needrm.size() > 0) ikdtree.Delete_Point_Boxes(cub_needrm); // 불필요한 포인트 제거
}

// 일반적인 포인트 클라우드 데이터를 처리하는 콜백 함수
void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    mtx_buffer.lock(); // 버퍼 보호를 위한 락 사용
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        // 라이다 타임스탬프가 이전 것보다 작으면, 데이터가 반복되었음을 의미
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear(); // 버퍼를 초기화
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI()); // 새로운 포인트 클라우드 생성
    p_pre->process(msg, ptr, 0); // 포인트 클라우드 데이터 전처리
    lidar_buffer.push_back(ptr); // 처리된 데이터를 버퍼에 저장
    time_buffer.push_back(msg->header.stamp.toSec()); // 타임스탬프도 버퍼에 저장
    last_timestamp_lidar = msg->header.stamp.toSec(); // 마지막 라이다 타임스탬프 갱신
    mtx_buffer.unlock(); // 락 해제
    sig_buffer.notify_all(); // 모든 대기 중인 쓰레드에 신호 전송
    first_lidar_scan_check = true; // 첫 라이다 스캔이 완료되었음을 체크
}

// 두 번째 라이다에 대한 포인트 클라우드 데이터를 처리하는 콜백 함수
void standard_pcl_cbk2(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    mtx_buffer.lock(); // 버퍼 보호를 위한 락 사용
    if (msg->header.stamp.toSec() < last_timestamp_lidar2)
    {
        // 라이다 타임스탬프가 이전 것보다 작으면, 데이터가 반복되었음을 의미
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear(); // 버퍼를 초기화
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI()); // 새로운 포인트 클라우드 생성
    p_pre->process(msg, ptr, 1); // 포인트 클라우드 데이터 전처리
    ptr->header.seq = 1; // 시퀀스 번호를 고정하는 트릭 (동기화를 위해)
    lidar_buffer.push_back(ptr); // 처리된 데이터를 버퍼에 저장
    time_buffer.push_back(msg->header.stamp.toSec()); // 타임스탬프도 버퍼에 저장
    last_timestamp_lidar2 = msg->header.stamp.toSec(); // 두 번째 라이다의 마지막 타임스탬프 갱신
    mtx_buffer.unlock(); // 락 해제
    sig_buffer.notify_all(); // 모든 대기 중인 쓰레드에 신호 전송
    first_lidar_scan_check = true; // 첫 라이다 스캔이 완료되었음을 체크
}

// Livox 라이다의 포인트 클라우드 데이터를 처리하는 콜백 함수
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) 
{
    mtx_buffer.lock(); // 버퍼 보호를 위한 락 사용
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        // 라이다 타임스탬프가 이전 것보다 작으면, 데이터가 반복되었음을 의미
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear(); // 버퍼를 초기화
    }
    last_timestamp_lidar = msg->header.stamp.toSec(); // 라이다의 마지막 타임스탬프 갱신
   
    PointCloudXYZI::Ptr ptr(new PointCloudXYZI()); // 새로운 포인트 클라우드 생성
    p_pre->process(msg, ptr, 0); // 포인트 클라우드 데이터 전처리
    lidar_buffer.push_back(ptr); // 처리된 데이터를 버퍼에 저장
    time_buffer.push_back(last_timestamp_lidar); // 타임스탬프도 버퍼에 저장
    mtx_buffer.unlock(); // 락 해제
    sig_buffer.notify_all(); // 모든 대기 중인 쓰레드에 신호 전송
    first_lidar_scan_check = true; // 첫 라이다 스캔이 완료되었음을 체크
}

// 두 번째 Livox 라이다의 포인트 클라우드 데이터를 처리하는 콜백 함수
void livox_pcl_cbk2(const livox_ros_driver::CustomMsg::ConstPtr &msg) 
{
    mtx_buffer.lock(); // 버퍼 보호를 위한 락 사용
    if (msg->header.stamp.toSec() < last_timestamp_lidar2)
    {
        // 라이다 타임스탬프가 이전 것보다 작으면, 데이터가 반복되었음을 의미
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear(); // 버퍼를 초기화
    }
    last_timestamp_lidar2 = msg->header.stamp.toSec(); // 두 번째 라이다의 마지막 타임스탬프 갱신
    
    PointCloudXYZI::Ptr ptr(new PointCloudXYZI()); // 새로운 포인트 클라우드 생성
    p_pre->process(msg, ptr, 1); // 포인트 클라우드 데이터 전처리
    ptr->header.seq = 1; // 시퀀스 번호를 고정하는 트릭 (동기화를 위해)
    lidar_buffer.push_back(ptr); // 처리된 데이터를 버퍼에 저장
    time_buffer.push_back(last_timestamp_lidar2); // 타임스탬프도 버퍼에 저장
    mtx_buffer.unlock(); // 락 해제
    sig_buffer.notify_all(); // 모든 대기 중인 쓰레드에 신호 전송
    first_lidar_scan_check = true; // 첫 라이다 스캔이 완료되었음을 체크
}
// IMU 데이터 콜백 함수
void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    // 첫 라이다 스캔이 아직 완료되지 않았다면 IMU 데이터를 무시함
    if (!first_lidar_scan_check) return;

    // IMU 데이터를 새로운 포인터로 복사
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
    double timestamp = msg->header.stamp.toSec(); // IMU 데이터의 타임스탬프를 가져옴

    // 버퍼 보호를 위해 락 사용
    mtx_buffer.lock();
    // 타임스탬프가 이전 IMU 데이터보다 작으면 데이터가 중복되었음을 의미
    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer"); // 경고 메시지 출력
        imu_buffer.clear(); // IMU 버퍼 초기화
    }
    last_timestamp_imu = timestamp; // 마지막 IMU 타임스탬프 갱신
    imu_buffer.push_back(msg); // 새로운 IMU 데이터를 버퍼에 추가
    mtx_buffer.unlock(); // 락 해제
    sig_buffer.notify_all(); // 대기 중인 쓰레드에 신호 전송
}



// 라이다 및 IMU 데이터를 동기화하는 함수
bool sync_packages(MeasureGroup &meas)
{
    // 라이다 또는 IMU 버퍼가 비어 있으면 동기화 실패
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** 라이다 스캔 데이터를 푸시 ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front(); // 라이다 데이터를 측정 그룹에 할당
        meas.lidar_beg_time = time_buffer.front(); // 라이다 시작 시간을 할당

        // 두 번째 라이다 데이터일 경우, 변환 적용 (LiDAR2와 LiDAR1 간의 좌표 변환)
        if (meas.lidar->header.seq == 1) // 트릭
        {
            pcl::transformPointCloud(*meas.lidar, *meas.lidar, LiDAR2_wrt_LiDAR1); // 좌표 변환
            current_lidar_num = 2; // 현재 라이다 번호를 2로 설정
            if (async_debug) cout << "\033[32;1mSecond LiDAR!" << "\033[0m" << endl;
        }
        else
        {
            current_lidar_num = 1; // 현재 라이다 번호를 1로 설정
            if (async_debug) cout << "\033[31;1mFirst LiDAR!" << "\033[0m" << endl;
        }

        // 포인트가 너무 적으면 경고 메시지 출력
        if (meas.lidar->points.size() <= 1) // 포인트가 너무 적음
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime; // 라이다 종료 시간 계산
            ROS_WARN("Too few input point cloud!\n"); // 경고 메시지 출력
        }
        // 라이다 스캔 시간이 평균 스캔 시간보다 짧으면 경고
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime; // 라이다 종료 시간 설정
        }
        else
        {
            scan_num ++; // 스캔 번호 증가
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000); // 라이다 종료 시간 계산
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num; // 평균 스캔 시간 갱신
        }

        meas.lidar_end_time = lidar_end_time; // 측정 그룹에 라이다 종료 시간 할당
        lidar_pushed = true; // 라이다 데이터가 푸시되었음을 표시
    }

    // 마지막 IMU 타임스탬프가 라이다 종료 시간보다 작으면 동기화 실패
    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** IMU 데이터를 푸시하고 버퍼에서 제거 ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec(); // 첫 번째 IMU 데이터의 타임스탬프 가져오기
    meas.imu.clear(); // 측정 그룹의 IMU 데이터를 초기화
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time)) // IMU 데이터가 라이다 종료 시간보다 작을 때까지
    {
        imu_time = imu_buffer.front()->header.stamp.toSec(); // 현재 IMU 데이터의 타임스탬프 갱신
        if(imu_time > lidar_end_time) break; // IMU 시간이 라이다 종료 시간보다 크면 루프 탈출
        meas.imu.push_back(imu_buffer.front()); // IMU 데이터를 측정 그룹에 추가
        imu_buffer.pop_front(); // 버퍼에서 IMU 데이터 제거
    }

    lidar_buffer.pop_front(); // 라이다 버퍼에서 데이터 제거
    time_buffer.pop_front(); // 타임 버퍼에서 데이터 제거
    lidar_pushed = false; // 라이다 푸시 완료 표시 초기화
    return true; // 동기화 성공
}

void map_incremental()
{
    PointVector PointToAdd; // 맵에 추가할 포인트 벡터
    PointVector PointNoNeedDownsample; // 다운샘플링이 필요 없는 포인트 벡터
    PointToAdd.reserve(feats_down_size); // 미리 벡터 크기를 예약하여 메모리 할당을 최적화
    PointNoNeedDownsample.reserve(feats_down_size); // 동일하게 다운샘플링이 필요 없는 포인트 벡터도 예약

    for (int i = 0; i < feats_down_size; i++) // 모든 다운샘플링 포인트에 대해 반복
    {
        /* 월드 좌표계로 변환 */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i])); // 바디 좌표계를 월드 좌표계로 변환
        
        /* 맵에 추가할 필요가 있는지 결정 */
        if (!Nearest_Points[i].empty()) // 근접 포인트가 있으면
        {
            const PointVector &points_near = Nearest_Points[i]; // 근접 포인트 가져오기
            bool need_add = true; // 맵에 추가할지 여부 플래그
            BoxPointType Box_of_Point; 
            PointType downsample_result, mid_point;

            // 다운샘플링 기준점 계산 (필터 크기를 기준으로 좌표를 정렬)
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_surf)*filter_size_surf + 0.5 * filter_size_surf;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_surf)*filter_size_surf + 0.5 * filter_size_surf;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_surf)*filter_size_surf + 0.5 * filter_size_surf;
            
            float dist  = calc_dist(feats_down_world->points[i], mid_point); // 포인트와 기준점 간의 거리 계산

            // 근접 포인트와 기준점의 거리 비교
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_surf && 
                fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_surf && 
                fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_surf)
            {
                PointNoNeedDownsample.push_back(feats_down_world->points[i]); // 다운샘플링 필요 없다고 판단된 포인트 추가
                continue; // 다음 포인트로 이동
            }

            // 근접 포인트들 중 기준점보다 가까운 포인트가 있는지 확인
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break; // 근접 포인트가 충분하지 않으면 중단
                if (calc_dist(points_near[readd_i], mid_point) < dist) // 근접 포인트 중 기준점보다 가까운 포인트가 있으면
                {
                    need_add = false; // 추가할 필요 없음
                    break;
                }
            }
            
            // 맵에 추가할 필요가 있다고 판단되면 추가
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else // 근접 포인트가 없는 경우 무조건 추가
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    // 맵에 포인트 추가 (다운샘플링이 필요한 포인트는 true로, 필요 없는 포인트는 false로)
    ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false);
    
    return;
}


void publish_frame_world(const ros::Publisher &pubLaserCloudFull, const ros::Publisher &pubLaserCloudFullTransFormed)
{
    if(scan_pub_en) // 스캔 퍼블리싱이 활성화되어 있을 때
    {
        // 퍼블리시할 포인트 클라우드 데이터를 준비. 자세한 포인트 클라우드 데이터를 퍼블리시할지, 다운샘플링된 데이터를 퍼블리시할지 선택
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size(); // 포인트 클라우드 크기 저장
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1)); // 월드 좌표계 포인트 클라우드 생성

        // 포인트 클라우드를 월드 좌표계로 변환
        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
        }

        // ROS 메시지로 변환하여 퍼블리시
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg); // PCL 포인트 클라우드를 ROS 메시지로 변환
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time); // 타임스탬프 설정
        laserCloudmsg.header.frame_id = map_frame; // 프레임 ID 설정
        pubLaserCloudFull.publish(laserCloudmsg); // 퍼블리시

        if (publish_tf_results) // 만약 변환된 좌표도 퍼블리시할 필요가 있을 때
        {
            PointCloudXYZI::Ptr laserCloudWorldTransFormed(new PointCloudXYZI(size, 1)); // 변환된 포인트 클라우드
            pcl::transformPointCloud(*laserCloudWorld, *laserCloudWorldTransFormed, LiDAR1_wrt_drone); // LiDAR와 드론 간의 변환을 적용
            sensor_msgs::PointCloud2 laserCloudmsg2;
            pcl::toROSMsg(*laserCloudWorldTransFormed, laserCloudmsg2); // PCL 포인트 클라우드를 ROS 메시지로 변환
            laserCloudmsg2.header.stamp = ros::Time().fromSec(lidar_end_time); // 타임스탬프 설정
            laserCloudmsg2.header.frame_id = map_frame; // 프레임 ID 설정
            pubLaserCloudFullTransFormed.publish(laserCloudmsg2); // 변환된 데이터를 퍼블리시
        }
    }

    /**************** 맵 저장 ****************/
    /* 1. 충분한 메모리가 있는지 확인
    /* 2. PCD 저장은 실시간 성능에 영향을 줄 수 있으므로 주의 */
    if (pcd_save_en) // PCD 저장 기능이 활성화되어 있을 때
    {
        int size = feats_undistort->points.size(); // 포인트 클라우드 크기 저장
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1)); // 월드 좌표계 포인트 클라우드 생성

        // 포인트 클라우드를 월드 좌표계로 변환
        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld; // PCD 대기열에 추가

        // 일정 간격으로 스캔 저장
        static int scan_wait_num = 0;
        scan_wait_num++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
        {
            pcd_index++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "현재 스캔이 /PCD/" << all_points_dir << " 에 저장되었습니다." << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save); // PCD 파일로 저장
            pcl_wait_save->clear(); // 저장 후 클라우드 클리어
            scan_wait_num = 0; // 대기 시간 초기화
        }
    }
}

void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    // 포인트 클라우드 크기
    int size = feats_undistort->points.size();
    // IMU 좌표계로 변환된 포인트 클라우드 생성
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    // 포인트 클라우드를 LiDAR 좌표계에서 IMU 좌표계로 변환
    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], &laserCloudIMUBody->points[i]);
    }

    // ROS 메시지로 변환 후 퍼블리시
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg); // PCL 포인트 클라우드를 ROS 메시지로 변환
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time); // 타임스탬프 설정
    laserCloudmsg.header.frame_id = "body"; // 프레임 ID를 IMU 좌표계로 설정
    pubLaserCloudFull_body.publish(laserCloudmsg); // 퍼블리시
}

void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    // ROS 메시지로 변환 후 퍼블리시
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap); // PCL 포인트 클라우드를 ROS 메시지로 변환
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time); // 타임스탬프 설정
    laserCloudMap.header.frame_id = map_frame; // 프레임 ID를 맵 좌표계로 설정
    pubLaserCloudMap.publish(laserCloudMap); // 퍼블리시
}

template<typename T>
void set_posestamp(T & out)
{
    // state_point에서 현재 위치와 자세 데이터를 가져와 출력 메시지에 설정
    out.pose.position.x = state_point.pos(0);  // x 좌표 설정
    out.pose.position.y = state_point.pos(1);  // y 좌표 설정
    out.pose.position.z = state_point.pos(2);  // z 좌표 설정
    out.pose.orientation.x = geoQuat.x;        // 쿼터니언 x 설정
    out.pose.orientation.y = geoQuat.y;        // 쿼터니언 y 설정
    out.pose.orientation.z = geoQuat.z;        // 쿼터니언 z 설정
    out.pose.orientation.w = geoQuat.w;        // 쿼터니언 w 설정
    return;
}

void publish_visionpose(const ros::Publisher &publisher)
{
    // 비전 포즈 메시지 생성
    geometry_msgs::PoseStamped msg_out_;
    msg_out_.header.frame_id = map_frame;  // 프레임 ID 설정
    msg_out_.header.stamp = ros::Time().fromSec(lidar_end_time);  // 타임스탬프 설정

    // 현재 상태에서 4x4 변환 행렬 생성
    Eigen::Matrix4d current_pose_eig_ = Eigen::Matrix4d::Identity();
    current_pose_eig_.block<3, 3>(0, 0) = state_point.rot.toRotationMatrix();  // 회전 행렬 설정
    current_pose_eig_.block<3, 1>(0, 3) = state_point.pos;  // 위치 설정

    // LiDAR 좌표계를 기준으로 변환된 비전 포즈 계산
    Eigen::Matrix4d tfed_vision_pose_eig_ = LiDAR1_wrt_drone * current_pose_eig_ * LiDAR1_wrt_drone.inverse(); // 변환 적용
    msg_out_.pose.position.x = tfed_vision_pose_eig_(0, 3);  // 변환된 x 좌표
    msg_out_.pose.position.y = tfed_vision_pose_eig_(1, 3);  // 변환된 y 좌표
    msg_out_.pose.position.z = tfed_vision_pose_eig_(2, 3);  // 변환된 z 좌표

    // 변환된 쿼터니언 설정
    Eigen::Quaterniond tfed_quat_(tfed_vision_pose_eig_.block<3, 3>(0, 0));
    msg_out_.pose.orientation.x = tfed_quat_.x();  // 쿼터니언 x
    msg_out_.pose.orientation.y = tfed_quat_.y();  // 쿼터니언 y
    msg_out_.pose.orientation.z = tfed_quat_.z();  // 쿼터니언 z
    msg_out_.pose.orientation.w = tfed_quat_.w();  // 쿼터니언 w

    // 퍼블리시
    publisher.publish(msg_out_);
    return;
}

void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    // odomAftMapped 메시지의 헤더에 프레임 ID를 설정 (지도 프레임 사용)
    odomAftMapped.header.frame_id = map_frame;

    // odomAftMapped 메시지의 자식 프레임 ID를 "body"로 설정
    odomAftMapped.child_frame_id = "body";

    // odomAftMapped 메시지의 시간 스탬프를 LiDAR 종료 시간으로 설정
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);

    // odomAftMapped 메시지의 위치 및 자세 데이터를 설정
    set_posestamp(odomAftMapped.pose);

    // odomAftMapped 메시지를 pubOdomAftMapped 퍼블리셔를 통해 전송
    pubOdomAftMapped.publish(odomAftMapped);

    // 칼만 필터(KF)에서 공분산 행렬 P를 가져옴
    auto P = kf.get_P();

    // 공분산 행렬 데이터를 odomAftMapped 메시지의 공분산 필드에 채워 넣음
    for (int i = 0; i < 6; i ++)
    {
        // 인덱스 i를 변환하여 사용 (i가 3보다 작으면 3을 더하고, 크면 3을 뺌)
        int k = i < 3 ? i + 3 : i - 3;

        // 공분산 행렬의 값을 odomAftMapped.pose.covariance에 할당
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    // odom에서 body로의 변환 브로드캐스터 생성
    static tf::TransformBroadcaster br_odom_to_body;

    // 변환 객체 생성
    tf::Transform transform;

    // odomAftMapped의 위치 데이터를 가져와 변환에 설정
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));

    // odomAftMapped의 자세(쿼터니언) 데이터를 가져와 변환에 설정
    tf::Quaternion q;
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);

    // 변환 객체에 회전값을 설정
    transform.setRotation(q);

    // odom에서 body로의 변환을 브로드캐스터를 통해 전송
    br_odom_to_body.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, map_frame, "body"));
}

void publish_path(const ros::Publisher pubPath)
{
    // geometry_msgs 타입의 PoseStamped 메시지 생성
    geometry_msgs::PoseStamped msg_body_pose;

    // msg_body_pose 메시지에 현재 위치 및 자세 데이터 설정
    set_posestamp(msg_body_pose);

    // LiDAR 종료 시간으로 시간 스탬프 설정
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);

    // 헤더 프레임 ID를 지도 프레임으로 설정
    msg_body_pose.header.frame_id = map_frame;

    /*** 경로 데이터가 너무 커지면 rvis가 충돌할 수 있음 ***/
    static int jjj = 0;  // 경로 업데이트를 제어하는 변수
    jjj++;  // 매 호출 시마다 1씩 증가

    // jjj 변수가 10의 배수일 때만 경로 추가 및 퍼블리시
    if (jjj % 10 == 0) 
    {
        // msg_body_pose를 path 메시지에 추가
        path.poses.push_back(msg_body_pose);

        // path 메시지를 pubPath 퍼블리셔를 통해 퍼블리시
        pubPath.publish(path);
    }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    // laserCloudOri 포인트 클라우드 데이터를 초기화 (비움)
    laserCloudOri->clear(); 

    // corr_normvect 데이터를 초기화 (비움)
    corr_normvect->clear(); 

    // 총 잔차 값을 0으로 초기화
    total_residual = 0.0; 

    /** 가장 가까운 표면 탐색 및 잔차 계산 **/
    #ifdef MP_EN
        // 멀티 프로세싱을 위한 스레드 수 설정
        omp_set_num_threads(MP_PROC_NUM);

        // OpenMP 병렬 처리 설정 (for 루프 병렬화)
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        // feats_down_body의 i번째 포인트를 참조
        PointType &point_body  = feats_down_body->points[i]; 

        // feats_down_world의 i번째 포인트를 참조
        PointType &point_world = feats_down_world->points[i]; 

        /* 포인트를 월드 프레임으로 변환 */
        // body 프레임에서의 포인트 좌표를 벡터로 저장
        V3D p_body(point_body.x, point_body.y, point_body.z);

        // 월드 프레임으로 변환 (회전 및 위치 변환 적용)
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

        // 변환된 월드 프레임 좌표를 point_world에 저장
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);

        // intensity 값을 body 포인트에서 world 포인트로 복사
        point_world.intensity = point_body.intensity;

        // NUM_MATCH_POINTS만큼의 최근접 포인트들의 제곱 거리 저장을 위한 벡터
        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        // 최근접 포인트들의 참조 변수
        auto &points_near = Nearest_Points[i];
                if (ekfom_data.converge)
        {
            /** 맵에서 가장 가까운 표면을 찾음 **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);

            // 포인트가 충분한지 및 거리가 5 이내인지 검사하여 표면 선택 여부 결정
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        // 선택된 표면이 없다면 다음 반복으로 넘어감
        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;  // 평면 방정식의 계수를 저장할 변수
        point_selected_surf[i] = false;

        // 근접 포인트들로부터 평면을 추정하고, 성공하면 잔차 계산
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            // 잔차가 충분히 작으면 표면 선택
            if (s > 0.9)
            {
                point_selected_surf[i] = true;

                // 노멀 벡터 업데이트
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);  // 잔차 값 저장
            }
        }
    }
    
    // 효과적인 피처의 개수 초기화
    effect_feat_num = 0;

    // localizability 계산을 위한 벡터 초기화
    localizability_vec = Eigen::Vector3d::Zero();

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            // 선택된 피처를 레이저 클라우드 및 노멀 벡터에 추가
            laserCloudOri->points[effect_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effect_feat_num] = normvec->points[i];

            // 총 잔차 값을 누적
            total_residual += res_last[i];
            effect_feat_num ++;

            // localizability 벡터에 노멀 벡터의 제곱 값을 추가
            localizability_vec += Eigen::Vector3d(normvec->points[i].x, normvec->points[i].y, normvec->points[i].z).array().square().matrix();
        }
    }

    // localizability 벡터의 제곱근을 계산
    localizability_vec = localizability_vec.cwiseSqrt();

    // 효과적인 피처가 하나도 없을 경우 경고 메시지 출력 및 함수 종료
    if (effect_feat_num < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }

    // 평균 잔차 계산
    res_mean_last = total_residual / effect_feat_num;
    
    /*** 측정 자코비안 행렬 H 및 측정 벡터 계산 ***/
    ekfom_data.h_x = MatrixXd::Zero(effect_feat_num, 12); //23
    ekfom_data.h.resize(effect_feat_num);

    for (int i = 0; i < effect_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];

        // 현재 포인트의 body 프레임 좌표 벡터
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);

        // body 프레임에서의 포인트의 스큐 대칭 행렬 생성
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);

        // 현재 포인트를 변환하여 세계 좌표계에서의 위치 계산
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;

        // 세계 좌표계에서의 포인트의 스큐 대칭 행렬 생성
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_this);

        /*** 가장 가까운 표면/코너의 노멀 벡터 가져오기 ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** 측정 자코비안 행렬 H 계산 ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);

        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** 측정 값: 가장 가까운 표면/코너까지의 거리 ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
}
int main(int argc, char** argv)
{
    // ROS 노드 초기화
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    // 임시 변수 선언 (외부 변환 행렬 및 벡터들)
    vector<double>       extrinT(3, 0.0);
    vector<double>       extrinR(9, 0.0);
    vector<double>       extrinT2(3, 0.0);
    vector<double>       extrinR2(9, 0.0);
    vector<double>       extrinT3(3, 0.0);
    vector<double>       extrinR3(9, 0.0);
    vector<double>       extrinT4(3, 0.0);
    vector<double>       extrinR4(9, 0.0);

    // 공통 설정 값 파라미터 로드
    nh.param<int>("common/max_iteration",NUM_MAX_ITERATIONS,4);
    nh.param<bool>("common/async_debug", async_debug, false);
    nh.param<bool>("common/multi_lidar", multi_lidar, true);
    nh.param<bool>("common/publish_tf_results", publish_tf_results, true);
    nh.param<string>("common/lid_topic",lid_topic,"/livox/lidar");
    nh.param<string>("common/lid_topic2",lid_topic2,"/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");
    nh.param<string>("common/map_frame", map_frame,"map");

    // 전처리 설정 값 파라미터 로드
    nh.param<double>("preprocess/filter_size_surf",filter_size_surf,0.5);
    nh.param<int>("preprocess/point_filter_num", p_pre->point_filter_num[0], 2);
    nh.param<int>("preprocess/point_filter_num2", p_pre->point_filter_num[1], 2);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type[0], AVIA);
    nh.param<int>("preprocess/lidar_type2", p_pre->lidar_type[1], AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS[0], 16);
    nh.param<int>("preprocess/scan_line2", p_pre->N_SCANS[1], 16);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE[0], 10);
    nh.param<int>("preprocess/scan_rate2", p_pre->SCAN_RATE[1], 10);
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit[0], US);
    nh.param<int>("preprocess/timestamp_unit2", p_pre->time_unit[1], US);
    nh.param<double>("preprocess/blind", p_pre->blind[0], 0.01);
    nh.param<double>("preprocess/blind2", p_pre->blind[1], 0.01);

    // 전처리 객체 설정
    p_pre->set();

    // 매핑 설정 값 파라미터 로드
    nh.param<double>("mapping/cube_side_length",cube_len,200.0);
    nh.param<float>("mapping/det_range",DET_RANGE,300.f);
    nh.param<double>("mapping/gyr_cov",gyr_cov,0.1);
    nh.param<double>("mapping/acc_cov",acc_cov,0.1);
    nh.param<double>("mapping/b_gyr_cov",b_gyr_cov,0.0001);
    nh.param<double>("mapping/b_acc_cov",b_acc_cov,0.0001);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<bool>("mapping/extrinsic_imu_to_lidars", extrinsic_imu_to_lidars, false);

    // 외부 변환 파라미터 로드 (LiDAR 간, LiDAR와 드론 간의 관계)
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_T2", extrinT2, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R2", extrinR2, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_T_L2_wrt_L1", extrinT3, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R_L2_wrt_L1", extrinR3, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_T_L1_wrt_drone", extrinT4, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R_L1_wrt_drone", extrinR4, vector<double>());

    // 퍼블리싱 관련 설정 값 파라미터 로드
    nh.param<bool>("publish/path_en",path_en, true);
    nh.param<bool>("publish/scan_publish_en",scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en",dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true);

    // PCD 저장 관련 설정 값 파라미터 로드
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    /*** 변수 초기화 ***/
    path.header.stamp    = ros::Time::now();  // 경로 메시지의 시간 스탬프를 현재 시간으로 설정
    path.header.frame_id = map_frame;         // 경로 메시지의 프레임 ID를 지도 프레임으로 설정
    memset(point_selected_surf, true, sizeof(point_selected_surf));  // point_selected_surf 배열을 true로 초기화
    memset(res_last, -1000.0f, sizeof(res_last));                    // res_last 배열을 -1000.0f로 초기화
    downSizeFilterSurf.setLeafSize(filter_size_surf, filter_size_surf, filter_size_surf);  // 다운샘플 필터 설정
    ikdtree.set_downsample_param(filter_size_surf);                  // KD 트리 다운샘플 파라미터 설정

    // IMU에 대한 LiDAR의 변환 행렬 및 벡터 설정
    V3D Lidar_T_wrt_IMU(Zero3d);
    M3D Lidar_R_wrt_IMU(Eye3d);
    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);                      // 변환 벡터 설정
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);                      // 변환 행렬 설정
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);          // IMU에 LiDAR 외부 변환 설정
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));              // 자이로스코프 공분산 설정
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));              // 가속도계 공분산 설정
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));   // 자이로스코프 바이어스 공분산 설정
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));   // 가속도계 바이어스 공분산 설정
    
    // 다중 LiDAR 설정 시
    if (multi_lidar)
    {
        if (extrinsic_imu_to_lidars)
        {
            Eigen::Matrix4d Lidar_wrt_IMU = Eigen::Matrix4d::Identity();  // LiDAR-IMU 변환 행렬 초기화
            Eigen::Matrix4d Lidar2_wrt_IMU = Eigen::Matrix4d::Identity(); // LiDAR2-IMU 변환 행렬 초기화
            V3D LiDAR2_T_wrt_IMU; LiDAR2_T_wrt_IMU << VEC_FROM_ARRAY(extrinT2);  // LiDAR2에 대한 변환 벡터 설정
            M3D LiDAR2_R_wrt_IMU; LiDAR2_R_wrt_IMU << MAT_FROM_ARRAY(extrinR2);  // LiDAR2에 대한 변환 행렬 설정
            Lidar_wrt_IMU.block<3,3>(0,0) = Lidar_R_wrt_IMU;                    // LiDAR-IMU 회전 행렬 설정
            Lidar_wrt_IMU.block<3,1>(0,3) = Lidar_T_wrt_IMU;                    // LiDAR-IMU 변환 벡터 설정
            Lidar2_wrt_IMU.block<3,3>(0,0) = LiDAR2_R_wrt_IMU;                  // LiDAR2-IMU 회전 행렬 설정
            Lidar2_wrt_IMU.block<3,1>(0,3) = LiDAR2_T_wrt_IMU;                  // LiDAR2-IMU 변환 벡터 설정
            LiDAR2_wrt_LiDAR1 = Lidar_wrt_IMU.inverse() * Lidar2_wrt_IMU;       // LiDAR2와 LiDAR1 간의 변환 계산
        }
        else
        {
            V3D LiDAR2_T_wrt_LiDAR1; LiDAR2_T_wrt_LiDAR1 << VEC_FROM_ARRAY(extrinT3);  // LiDAR2-LiDAR1 변환 벡터 설정
            M3D Lidar2_R_wrt_LiDAR1; Lidar2_R_wrt_LiDAR1 << MAT_FROM_ARRAY(extrinR3);  // LiDAR2-LiDAR1 회전 행렬 설정
            LiDAR2_wrt_LiDAR1.block<3,3>(0,0) = Lidar2_R_wrt_LiDAR1;                  // LiDAR2-LiDAR1 회전 설정
            LiDAR2_wrt_LiDAR1.block<3,1>(0,3) = LiDAR2_T_wrt_LiDAR1;                  // LiDAR2-LiDAR1 변환 설정
        }
        cout << "\033[32;1mMulti LiDAR on!" << endl; // 다중 LiDAR 활성화 메시지 출력
        cout << "lidar_type[0]: " << p_pre->lidar_type[0] << ", " << "lidar_type[1]: " << p_pre->lidar_type[1] << endl << endl;
        cout << "L2 wrt L1 TF: " << endl << LiDAR2_wrt_LiDAR1 << "\033[0m" << endl << endl;        
    }

    // 변환 퍼블리싱 활성화 시
    if (publish_tf_results)
    {
        V3D LiDAR1_T_wrt_drone; LiDAR1_T_wrt_drone << VEC_FROM_ARRAY(extrinT4);  // LiDAR1-드론 변환 벡터 설정
        M3D LiDAR2_R_wrt_drone; LiDAR2_R_wrt_drone << MAT_FROM_ARRAY(extrinR4);  // LiDAR2-드론 회전 행렬 설정
        LiDAR1_wrt_drone.block<3,3>(0,0) = LiDAR2_R_wrt_drone;                   // LiDAR1-드론 회전 설정
        LiDAR1_wrt_drone.block<3,1>(0,3) = LiDAR1_T_wrt_drone;                   // LiDAR1-드론 변환 설정
        cout << "\033[32;1mLiDAR wrt Drone:" << endl;
        cout << LiDAR1_wrt_drone << "\033[0m" << endl << endl;
    }

    // 칼만 필터 초기화
    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);  // epsi 배열 초기화
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);  // 칼만 필터 초기화

    /*** ROS 구독 초기화 ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type[0] == AVIA ? \
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 200000, standard_pcl_cbk);  // 첫 번째 LiDAR 데이터 구독 설정
    ros::Subscriber sub_pcl2;
    if (multi_lidar)
    {
        sub_pcl2 = p_pre->lidar_type[1] == AVIA ? \
            nh.subscribe(lid_topic2, 200000, livox_pcl_cbk2) : \
            nh.subscribe(lid_topic2, 200000, standard_pcl_cbk2);  // 두 번째 LiDAR 데이터 구독 설정
    }
    
    // IMU 데이터를 구독
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);

    // 다양한 퍼블리셔를 설정하여 관련 데이터를 퍼블리시
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered", 100000); // 전체 LiDAR 포인트 클라우드 퍼블리셔

    ros::Publisher pubLaserCloudFullTransformed = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_tf", 100000); // 변환된 LiDAR 포인트 클라우드 퍼블리셔

    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_body", 100000); // Body 프레임에서의 LiDAR 포인트 클라우드 퍼블리셔

    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
            ("/Laser_map", 100000); // 매핑된 LiDAR 포인트 클라우드 퍼블리셔

    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> 
            ("/Odometry", 100000); // 매핑 이후의 Odometry 데이터 퍼블리셔

    ros::Publisher pubMavrosVisionPose = nh.advertise<geometry_msgs::PoseStamped> 
            ("/mavros/vision_pose/pose", 100000); // MAVROS 비전 포즈 퍼블리셔

    ros::Publisher pubPath = nh.advertise<nav_msgs::Path> 
            ("/path", 100000); // 경로 데이터를 퍼블리시하는 퍼블리셔

    ros::Publisher pubCaclTime = nh.advertise<std_msgs::Float32> 
            ("/calc_time", 100000); // 계산 시간 데이터를 퍼블리시하는 퍼블리셔

    ros::Publisher pubPointNum = nh.advertise<std_msgs::Float32> 
            ("/point_number", 100000); // 포인트 클라우드 포인트 수 퍼블리셔

    ros::Publisher pubLocalizabilityX = nh.advertise<std_msgs::Float32> 
            ("/localizability_x", 100000); // X축에서의 로컬라이저빌리티 퍼블리셔

    ros::Publisher pubLocalizabilityY = nh.advertise<std_msgs::Float32> 
            ("/localizability_y", 100000); // Y축에서의 로컬라이저빌리티 퍼블리셔

    ros::Publisher pubLocalizabilityZ = nh.advertise<std_msgs::Float32> 
            ("/localizability_z", 100000); // Z축에서의 로컬라이저빌리티 퍼블리셔

// SIGINT 신호를 처리하기 위한 핸들러 설정
signal(SIGINT, SigHandle);

// ROS 루프 주기를 5000Hz로 설정
ros::Rate rate(5000);
bool status = ros::ok();  // ROS 상태 확인

    while (status)
    {
        // 종료 플래그가 설정되면 루프 탈출
        if (flg_exit) break;

        // ROS 메시지 콜백 처리
        ros::spinOnce();

        // 패키지 동기화가 성공하면 처리 시작
        if(sync_packages(Measures)) 
        {
            // 현재 시간을 기록 (고해상도 타이머 사용)
            high_resolution_clock::time_point t1 = high_resolution_clock::now();

            // IMU 데이터를 처리하여 상태 업데이트
            p_imu->Process(Measures, kf, feats_undistort, false, 1);  // IMU 처리, false로 동기화 모드 비활성화
            state_point = kf.get_x();  // 필터링된 상태 정보를 가져옴

            // LiDAR의 현재 위치를 계산 (상태 정보에서 변환)
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            // 필터링된 피처 포인트가 없거나 null이면 스캔을 건너뜀
            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            /*** LiDAR FOV에서 맵을 분할 ***/
            lasermap_fov_segment();

            /*** 스캔 내의 피처 포인트 다운샘플링 ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);  // 다운샘플링된 피처 포인트를 feats_down_body에 저장
            feats_down_size = feats_down_body->points.size();  // 다운샘플링된 포인트 수를 저장


            /*** 맵 KD 트리 초기화 ***/
    if (multi_lidar)
    {
        // LiDAR 1 또는 LiDAR 2의 KD 트리가 초기화되지 않은 경우
        if(ikdtree.Root_Node == nullptr || !lidar1_ikd_init || !lidar2_ikd_init)
        {
            if(feats_down_size > 5)
            {
                // 다운샘플링된 포인트 클라우드 크기 조정 및 변환
                feats_down_world->resize(feats_down_size);
                for (int i = 0; i < feats_down_size; i++)
                {
                    pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                }

                // KD 트리에 포인트 추가
                ikdtree.Add_Points(feats_down_world->points, true);

                // 현재 LiDAR가 1번이면 LiDAR 1 KD 트리 초기화 완료 플래그 설정
                if (current_lidar_num == 1)
                {
                    lidar1_ikd_init = true;
                }
                // 현재 LiDAR가 2번이면 LiDAR 2 KD 트리 초기화 완료 플래그 설정
                else if (current_lidar_num == 2)
                {
                    lidar2_ikd_init = true;
                }
            }
            continue;  // 루프 계속
        }
    }
    else if(ikdtree.Root_Node == nullptr)  // 단일 LiDAR인 경우 KD 트리 초기화 확인
    {
        if(feats_down_size > 5)
        {
            // 포인트 변환 후 KD 트리에 추가
            feats_down_world->resize(feats_down_size);
            for (int i = 0; i < feats_down_size; i++)
            {
                pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
            }
            ikdtree.Add_Points(feats_down_world->points, true);
        }
        continue;
    }

    /*** ICP 및 반복 칼만 필터 업데이트 ***/
    if (feats_down_size < 5)  // 포인트 수가 5 미만인 경우 스캔 건너뜀
    {
        ROS_WARN("No point, skip this scan!\n");
        continue;
    }

    // 노멀 벡터 및 변환된 포인트, 근접 포인트 메모리 크기 조정
    normvec->resize(feats_down_size);
    feats_down_world->resize(feats_down_size);
    Nearest_Points.resize(feats_down_size);

    /*** 반복 상태 추정 업데이트 ***/
    double solve_H_time = 0;
    kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);  // 칼만 필터 업데이트
    state_point = kf.get_x();  // 상태 값 업데이트
    pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;  // LiDAR 위치 계산
    geoQuat.x = state_point.rot.coeffs()[0];  // 회전 값 갱신
    geoQuat.y = state_point.rot.coeffs()[1];
    geoQuat.z = state_point.rot.coeffs()[2];
    geoQuat.w = state_point.rot.coeffs()[3];

    /******* 오도메트리 퍼블리시 *******/
    if (publish_tf_results) publish_visionpose(pubMavrosVisionPose);  // MAVROS 비전 포즈 퍼블리시
    publish_odometry(pubOdomAftMapped);  // 오도메트리 퍼블리시

    /*** 맵 KD 트리에 피처 포인트 추가 ***/
    map_incremental();  // 점진적으로 맵 업데이트

    if(0) // 맵 포인트를 확인하려면 "if(1)"로 변경
    {
        PointVector().swap(ikdtree.PCL_Storage);  // KD 트리의 포인트 저장소 교체
        ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);  // KD 트리 평탄화
        featsFromMap->clear();  // 맵 피처 클리어
        featsFromMap->points = ikdtree.PCL_Storage;  // 피처 포인트 복사
    }

    /******* 포인트 퍼블리시 *******/
    if (path_en)                         publish_path(pubPath);  // 경로 퍼블리시
    if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFull, pubLaserCloudFullTransformed);  // 월드 프레임 퍼블리시
    if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);  // 바디 프레임 퍼블리시

    // 실행 시간 계산
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(t2 - t1).count() / 1000.0;
    std_msgs::Float32 calc_time;
    calc_time.data = duration;  // 계산 시간 퍼블리시
    pubCaclTime.publish(calc_time);

    // 포인트 수 퍼블리시
    std_msgs::Float32 point_num;
    point_num.data = feats_down_size;
    pubPointNum.publish(point_num);

    // 로컬라이저빌리티 퍼블리시
    std_msgs::Float32 localizability_x, localizability_y, localizability_z;
    localizability_x.data = localizability_vec(0);
    localizability_y.data = localizability_vec(1);
    localizability_z.data = localizability_vec(2);
    pubLocalizabilityX.publish(localizability_x);
    pubLocalizabilityY.publish(localizability_y);
    pubLocalizabilityZ.publish(localizability_z);
}
    
    status = ros::ok();  // ROS 상태 갱신
    rate.sleep();  // 주기적인 슬립 호출
}
    /**************** 맵 저장 ****************/
    /* 1. 충분한 메모리가 있는지 확인하십시오.
    /* 2. PCD 저장은 실시간 성능에 큰 영향을 미칠 수 있습니다. **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        // 파일 이름 설정
        string file_name = string("scans.pcd");

        // 파일 경로 설정
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);

        // PCD 파일 작성자 객체 생성
        pcl::PCDWriter pcd_writer;

        // 파일 저장 경로 출력
        cout << "current scan saved to /PCD/" << file_name << endl;

        // 포인트 클라우드를 바이너리 형식으로 PCD 파일로 저장
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }
    
    return 0;  // 프로그램 종료
}



