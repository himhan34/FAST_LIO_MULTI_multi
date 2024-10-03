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

/// C++ 표준 라이브러리 헤더
#include <mutex>  // 상호 배제를 위한 mutex 라이브러리 포함
#include <cmath>  // 수학 함수 사용을 위한 라이브러리 포함
#include <csignal>  // 신호 처리를 위한 라이브러리 포함
#include <unistd.h>  // 유닉스 표준 시스템 호출 라이브러리 포함
#include <condition_variable>  // 조건 변수를 사용하기 위한 라이브러리 포함

/// 모듈 헤더
#include <omp.h>  // OpenMP를 사용한 병렬 처리를 위한 라이브러리 포함

/// Eigen 라이브러리
#include <Eigen/Core>  // Eigen의 선형대수 관련 기능 포함

/// ROS 라이브러리
#include <ros/ros.h>  // ROS의 기본 기능을 위한 라이브러리 포함
#include <geometry_msgs/Vector3.h>  // ROS의 3차원 벡터 메시지 포함
#include <geometry_msgs/PoseStamped.h>  // ROS의 위치 메시지 포함
#include <nav_msgs/Odometry.h>  // ROS의 위치 추정 메시지 포함
#include <nav_msgs/Path.h>  // ROS의 경로 메시지 포함
#include <sensor_msgs/PointCloud2.h>  // ROS의 포인트 클라우드 메시지 포함
#include <tf/transform_datatypes.h>  // TF 변환 데이터 유형 포함
#include <tf/transform_broadcaster.h>  // TF 변환 브로드캐스터 포함
#include <livox_ros_driver/CustomMsg.h>  // Livox LiDAR의 커스텀 메시지 포함

/// PCL(Point Cloud Library)
#include <pcl/point_cloud.h>  // PCL 포인트 클라우드 타입 포함
#include <pcl/point_types.h>  // PCL 포인트 타입 정의 포함
#include <pcl/common/transforms.h> // 포인트 클라우드 변환을 위한 함수 포함
#include <pcl/filters/voxel_grid.h>  // PCL의 Voxel Grid 필터 포함
#include <pcl_conversions/pcl_conversions.h>  // PCL과 ROS 메시지 간 변환 포함
#include <pcl/io/pcd_io.h>  // PCD 파일 입출력을 위한 라이브러리 포함

/// 이 패키지 관련 헤더
#include "so3_math.h"  // SO3 관련 수학 연산을 위한 사용자 정의 헤더 포함
#include "IMU_Processing.hpp"  // IMU 데이터 처리 관련 사용자 정의 헤더 포함
#include "preprocess.h"  // 데이터 전처리 관련 헤더 포함
#include <ikd-Tree/ikd_Tree.h>  // ikd-Tree 구조를 사용하기 위한 헤더 포함
#include <chrono>  // 시간 측정을 위한 chrono 라이브러리 포함
#include <std_msgs/Float32.h>  // ROS의 Float32 메시지 포함

using namespace std::chrono;  // chrono 네임스페이스 사용

#define LASER_POINT_COV     (0.001)  // 레이저 포인트의 공분산 값을 0.001로 정의

/**************************/
// 전역 변수 설정
bool pcd_save_en = false, extrinsic_est_en = true, path_en = true;  // PCD 저장, 외부 파라미터 추정, 경로 활성화 여부
float res_last[100000] = {0.0};  // 마지막 잔차 값을 저장하는 배열
float DET_RANGE = 300.0f;  // 탐지 범위 설정
const float MOV_THRESHOLD = 1.5f;  // 움직임 임계값 설정

string lid_topic, lid_topic2, imu_topic, map_frame = "map";  // LiDAR와 IMU 토픽, 맵 프레임 이름
bool multi_lidar = false, async_debug = false, publish_tf_results = false, bundle_enabled = false;  
// 다중 LiDAR 사용 여부, 비동기 디버그, TF 결과 퍼블리싱, 번들 활성화 여부
bool extrinsic_imu_to_lidars = false;  // IMU와 LiDAR 간의 외부 파라미터 추정 활성화 여부

int voxelized_pt_num_thres = 100, bundle_enabled_tic = 10000, bundle_enabled_tic_thres = 10;  
// Voxel 그리드에서 포인트 수 임계값, 번들 활성화 주기 및 임계값

double effect_pt_num_ratio_thres = 0.4;  // 효과적인 포인트 수 비율 임계값

double res_mean_last = 0.05, total_residual = 0.0;  // 마지막 평균 잔차, 총 잔차
double last_timestamp_lidar = 0, last_timestamp_lidar2 = 0, last_timestamp_imu = -1.0;  
// 마지막 LiDAR, IMU 타임스탬프
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;  
// 자이로스코프, 가속도계 공분산 및 바이어스 공분산 값
double filter_size_surf = 0;  // 표면 필터 크기
double cube_len = 0, lidar_end_time = 0, lidar_end_time2 = 0, first_lidar_time = 0.0;  
// 큐브 길이, LiDAR 종료 시간 및 첫 번째 LiDAR 시간
double publish_lidar_time = 0.0;  // LiDAR 퍼블리시 시간
int    effect_feat_num = 0;  // 효과적인 특징 포인트 수
int    feats_down_size = 0, NUM_MAX_ITERATIONS = 0, pcd_save_interval = -1, pcd_index = 0;  
// 다운샘플링된 특징 포인트 수, 최대 반복 횟수, PCD 저장 간격, PCD 인덱스

bool   point_selected_surf[100000] = {0};  // 선택된 표면 포인트 여부 배열
bool   lidar_pushed = false, flg_exit = false;  // LiDAR 데이터가 입력되었는지 여부 및 종료 플래그
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;  
// 스캔 퍼블리시, 밀집 퍼블리시, 스캔 바디 퍼블리시 여부

vector<BoxPointType> cub_needrm;  // 삭제가 필요한 박스 포인트들을 저장하는 벡터
vector<PointVector>  Nearest_Points;  // 가장 가까운 포인트들을 저장하는 벡터
deque<double>                     time_buffer;  // 첫 번째 LiDAR의 타임스탬프를 저장하는 덱
deque<double>                     time_buffer2;  // 두 번째 LiDAR의 타임스탬프를 저장하는 덱
deque<PointCloudXYZI::Ptr>        lidar_buffer;  // 첫 번째 LiDAR의 포인트 클라우드를 저장하는 덱
deque<PointCloudXYZI::Ptr>        lidar_buffer2;  // 두 번째 LiDAR의 포인트 클라우드를 저장하는 덱
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;  // IMU 데이터를 저장하는 덱
mutex mtx_buffer;  // 버퍼 보호를 위한 상호 배제 뮤텍스
condition_variable sig_buffer;  // 버퍼의 조건 변수를 사용한 동기화

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());  // 맵에서 얻은 특징 포인트들을 저장하는 포인터
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());  // 왜곡을 보정한 특징 포인트들을 저장하는 포인터
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());  // 바디 프레임에서 다운샘플링된 특징 포인트들 저장
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());  // 월드 프레임에서 다운샘플링된 특징 포인트들 저장
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));  // 법선 벡터를 저장하는 포인트 클라우드 포인터 (크기 100000)
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));  // 원래 레이저 클라우드를 저장하는 포인트 클라우드 포인터
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));  // 보정된 법선 벡터를 저장하는 포인트 클라우드 포인터
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());  // 저장을 기다리는 포인트 클라우드를 저장하는 포인터

pcl::VoxelGrid<PointType> downSizeFilterSurf;  // Voxel Grid 필터를 사용하여 포인트 클라우드 다운샘플링
KD_TREE<PointType> ikdtree;  // KD 트리 구조로 포인트 클라우드를 저장하고 검색하기 위한 트리
shared_ptr<Preprocess> p_pre(new Preprocess());  // 전처리 객체를 위한 스마트 포인터
shared_ptr<ImuProcess> p_imu(new ImuProcess());  // IMU 처리 객체를 위한 스마트 포인터

Eigen::Matrix4d LiDAR2_wrt_LiDAR1 = Eigen::Matrix4d::Identity();  // LiDAR 1에 대한 LiDAR 2의 변환 행렬 (단위 행렬로 초기화)
Eigen::Matrix4d LiDAR1_wrt_drone = Eigen::Matrix4d::Identity();  // 드론에 대한 LiDAR 1의 변환 행렬 (단위 행렬로 초기화)


/*** EKF inputs and output ***/
MeasureGroup Measures;  // 측정 그룹 데이터를 저장하는 객체
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;  // 상태 예측을 위한 확장 칼만 필터 객체
state_ikfom state_point;  // 상태 포인트 데이터를 저장하는 객체
vect3 pos_lid;  // LiDAR 위치를 저장하는 벡터

nav_msgs::Path path;  // ROS 경로 메시지 객체
nav_msgs::Odometry odomAftMapped;  // 매핑 후의 위치 정보를 저장하는 ROS 위치 추정 메시지 객체
geometry_msgs::Quaternion geoQuat;  // 쿼터니언을 저장하는 객체

BoxPointType LocalMap_Points;  // 지역 맵의 포인트들을 저장하는 객체
bool Localmap_Initialized = false;  // 지역 맵이 초기화되었는지 여부
bool first_lidar_scan_check = false;  // 첫 번째 LiDAR 스캔 여부 확인
bool lidar1_ikd_init = false, lidar2_ikd_init = false;  // 첫 번째와 두 번째 LiDAR의 KD-트리 초기화 여부
int current_lidar_num = 1;  // 현재 사용 중인 LiDAR 번호
double lidar_mean_scantime = 0.0;  // 첫 번째 LiDAR의 평균 스캔 시간
double lidar_mean_scantime2 = 0.0;  // 두 번째 LiDAR의 평균 스캔 시간
int    scan_num = 0;  // 첫 번째 LiDAR의 스캔 횟수
int    scan_num2 = 0;  // 두 번째 LiDAR의 스캔 횟수
Eigen::Vector3d localizability_vec = Eigen::Vector3d::Zero();  // 위치 추정 가능성을 나타내는 벡터 (초기값 0)

void SigHandle(int sig)
{
    flg_exit = true;  // 종료 플래그를 true로 설정
    ROS_WARN("catch sig %d", sig);  // 잡힌 신호를 출력
    sig_buffer.notify_all();  // 버퍼 조건 변수에 신호를 보냄
}

void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);  // 입력 포인트의 바디 좌표계에서의 위치
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);  
    // LiDAR 위치를 월드 좌표계로 변환

    po->x = p_global(0);  // 변환된 X 좌표
    po->y = p_global(1);  // 변환된 Y 좌표
    po->z = p_global(2);  // 변환된 Z 좌표
    po->intensity = pi->intensity;  // 원본 포인트의 intensity 값을 복사
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);  // 입력 포인트의 바디 좌표계에서의 위치
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);  
    // LiDAR 위치를 월드 좌표계로 변환

    po[0] = p_global(0);  // 변환된 X 좌표
    po[1] = p_global(1);  // 변환된 Y 좌표
    po[2] = p_global(2);  // 변환된 Z 좌표
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);  // 입력 포인트의 바디 좌표계에서의 위치
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);  
    // LiDAR 포인트를 월드 좌표계로 변환

    po->x = p_global(0);  // 변환된 X 좌표
    po->y = p_global(1);  // 변환된 Y 좌표
    po->z = p_global(2);  // 변환된 Z 좌표
    po->intensity = pi->intensity;  // 원본 포인트의 intensity 값을 복사
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);  // 입력 포인트의 LiDAR 좌표계에서의 위치
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);  
    // LiDAR 좌표계를 IMU 좌표계로 변환

    po->x = p_body_imu(0);  // 변환된 X 좌표
    po->y = p_body_imu(1);  // 변환된 Y 좌표
    po->z = p_body_imu(2);  // 변환된 Z 좌표
    po->intensity = pi->intensity;  // 원본 포인트의 intensity 값을 복사
}


void lasermap_fov_segment()
{
    cub_needrm.clear();  // 제거할 큐브 목록을 초기화
    V3D pos_LiD = pos_lid;  // 현재 LiDAR 위치
    if (!Localmap_Initialized){  // 지역 맵이 초기화되지 않은 경우
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;  // 지역 맵 최소 좌표 설정
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;  // 지역 맵 최대 좌표 설정
        }
        Localmap_Initialized = true;  // 지역 맵 초기화 완료
        return;
    }
    float dist_to_map_edge[3][2];  // 맵 가장자리까지의 거리
    bool need_move = false;  // 지역 맵을 이동할 필요 여부
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);  // 지역 맵 최소 좌표까지의 거리
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);  // 지역 맵 최대 좌표까지의 거리
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;  
        // 만약 지역 맵 가장자리에 너무 가까우면 이동 필요
    }
    if (!need_move) return;  // 이동할 필요가 없으면 함수 종료
    BoxPointType New_LocalMap_Points, tmp_boxpoints;  // 새로운 지역 맵 포인트와 임시 박스 포인트
    New_LocalMap_Points = LocalMap_Points;  // 새로운 지역 맵 포인트에 현재 포인트 할당
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));  
    // 이동 거리 계산
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;  // 임시 박스 포인트에 현재 지역 맵 할당
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){  // 맵 최소 좌표에 가까운 경우
            New_LocalMap_Points.vertex_max[i] -= mov_dist;  // 새로운 맵 최대 좌표를 이동
            New_LocalMap_Points.vertex_min[i] -= mov_dist;  // 새로운 맵 최소 좌표를 이동
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;  // 이동한 최소 좌표 설정
            cub_needrm.push_back(tmp_boxpoints);  // 제거할 박스 목록에 추가
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){  // 맵 최대 좌표에 가까운 경우
            New_LocalMap_Points.vertex_max[i] += mov_dist;  // 새로운 맵 최대 좌표를 이동
            New_LocalMap_Points.vertex_min[i] += mov_dist;  // 새로운 맵 최소 좌표를 이동
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;  // 이동한 최대 좌표 설정
            cub_needrm.push_back(tmp_boxpoints);  // 제거할 박스 목록에 추가
        }
    }
    LocalMap_Points = New_LocalMap_Points;  // 새로운 지역 맵 포인트를 현재 지역 맵에 할당

    if(cub_needrm.size() > 0) ikdtree.Delete_Point_Boxes(cub_needrm);  // 제거할 박스가 있으면 KD-트리에서 삭제
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    mtx_buffer.lock();  // 버퍼 보호를 위한 락
    if (msg->header.stamp.toSec() < last_timestamp_lidar)  // LiDAR의 타임스탬프가 이전보다 작으면
    {
        ROS_ERROR("lidar loop back, clear buffer");  // LiDAR 루프백 에러 출력
        lidar_buffer.clear();  // LiDAR 버퍼 초기화
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());  // 새로운 포인트 클라우드 포인터 생성
    p_pre->process(msg, ptr, 0);  // 메시지 처리
    lidar_buffer.push_back(ptr);  // LiDAR 버퍼에 추가
    time_buffer.push_back(msg->header.stamp.toSec());  // 시간 버퍼에 타임스탬프 추가
    last_timestamp_lidar = msg->header.stamp.toSec();  // 마지막 타임스탬프 업데이트
    mtx_buffer.unlock();  // 락 해제
    sig_buffer.notify_all();  // 조건 변수에 신호 보내기
    first_lidar_scan_check = true;  // 첫 번째 LiDAR 스캔 체크
}

void standard_pcl_cbk2(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    mtx_buffer.lock();  // 버퍼 보호를 위한 락
    if (msg->header.stamp.toSec() < last_timestamp_lidar2)  // 두 번째 LiDAR의 타임스탬프가 이전보다 작으면
    {
        ROS_ERROR("lidar loop back, clear buffer");  // LiDAR 루프백 에러 출력
        lidar_buffer2.clear();  // 두 번째 LiDAR 버퍼 초기화
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());  // 새로운 포인트 클라우드 포인터 생성
    p_pre->process(msg, ptr, 1);  // 메시지 처리
    ptr->header.seq = 1;  // 임시로 시퀀스 설정 (임시 해결책)
    lidar_buffer2.push_back(ptr);  // 두 번째 LiDAR 버퍼에 추가
    time_buffer2.push_back(msg->header.stamp.toSec());  // 두 번째 시간 버퍼에 타임스탬프 추가
    last_timestamp_lidar2 = msg->header.stamp.toSec();  // 마지막 타임스탬프 업데이트
    mtx_buffer.unlock();  // 락 해제
    sig_buffer.notify_all();  // 조건 변수에 신호 보내기
    first_lidar_scan_check = true;  // 첫 번째 LiDAR 스캔 체크
}

void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) 
{
    mtx_buffer.lock();  // 버퍼 보호를 위한 락
    if (msg->header.stamp.toSec() < last_timestamp_lidar)  // LiDAR의 타임스탬프가 이전보다 작으면
    {
        ROS_ERROR("lidar loop back, clear buffer");  // LiDAR 루프백 에러 출력
        lidar_buffer.clear();  // LiDAR 버퍼 초기화
    }
    last_timestamp_lidar = msg->header.stamp.toSec();  // 마지막 타임스탬프 업데이트

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());  // 새로운 포인트 클라우드 포인터 생성
    p_pre->process(msg, ptr, 0);  // 메시지 처리
    lidar_buffer.push_back(ptr);  // LiDAR 버퍼에 추가
    time_buffer.push_back(last_timestamp_lidar);  // 시간 버퍼에 타임스탬프 추가
    
    mtx_buffer.unlock();  // 락 해제
    sig_buffer.notify_all();  // 조건 변수에 신호 보내기
    first_lidar_scan_check = true;  // 첫 번째 LiDAR 스캔 체크
}

void livox_pcl_cbk2(const livox_ros_driver::CustomMsg::ConstPtr &msg) 
{
    mtx_buffer.lock();  // 버퍼 보호를 위한 락
    if (msg->header.stamp.toSec() < last_timestamp_lidar2)  // 두 번째 LiDAR의 타임스탬프가 이전보다 작으면
    {
        ROS_ERROR("lidar loop back, clear buffer");  // LiDAR 루프백 에러 출력
        lidar_buffer2.clear();  // 두 번째 LiDAR 버퍼 초기화
    }
    last_timestamp_lidar2 = msg->header.stamp.toSec();  // 마지막 타임스탬프 업데이트
    
    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());  // 새로운 포인트 클라우드 포인터 생성
    p_pre->process(msg, ptr, 1);  // 메시지 처리
    ptr->header.seq = 1;  // 임시로 시퀀스 설정 (임시 해결책)
    lidar_buffer2.push_back(ptr);  // 두 번째 LiDAR 버퍼에 추가
    time_buffer2.push_back(last_timestamp_lidar2);  // 두 번째 시간 버퍼에 타임스탬프 추가
    
    mtx_buffer.unlock();  // 락 해제
    sig_buffer.notify_all();  // 조건 변수에 신호 보내기
    first_lidar_scan_check = true;  // 첫 번째 LiDAR 스캔 체크
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    if (!first_lidar_scan_check) return;  // 첫 번째 LiDAR 스캔이 없으면 IMU 입력만 스택되도록 방지

    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));  // IMU 데이터를 새로운 메시지로 복사
    double timestamp = msg->header.stamp.toSec();  // 타임스탬프 가져오기

    mtx_buffer.lock();  // 버퍼 보호를 위한 락

    if (timestamp < last_timestamp_imu)  // IMU 타임스탬프가 이전보다 작으면
    {
        ROS_WARN("imu loop back, clear buffer");  // IMU 루프백 경고 출력
        imu_buffer.clear();  // IMU 버퍼 초기화
    }

    last_timestamp_imu = timestamp;  // 마지막 IMU 타임스탬프 업데이트

    imu_buffer.push_back(msg);  // IMU 데이터를 버퍼에 추가
    mtx_buffer.unlock();  // 락 해제
    sig_buffer.notify_all();  // 조건 변수에 신호 보내기
}

bool sync_packages_async(MeasureGroup &meas)
{
    if ( (lidar_buffer.empty() && lidar_buffer2.empty()) || imu_buffer.empty()) {
        return false;  // LiDAR와 IMU 버퍼가 비어 있으면 false 반환
    }

    int lidar_use_num_now_ = 0;  // 현재 사용 중인 LiDAR 번호 초기화
    if (!lidar_buffer.empty() && !lidar_buffer2.empty()) // 두 개의 LiDAR 모두 데이터가 있으면 더 오래된 것을 사용
    {
        // LiDAR 1이 더 오래된 경우
        if (time_buffer.front() < time_buffer2.front())
        {
            lidar_use_num_now_ = 1;
        }
        // LiDAR 2가 더 오래된 경우
        else
        {
            lidar_use_num_now_ = 2;
        }    
    }
    else
    {
        if (!lidar_buffer.empty())
        {
            lidar_use_num_now_ = 1;  // LiDAR 1만 데이터가 있으면 LiDAR 1 사용
        }
        else if (!lidar_buffer2.empty())
        {
            lidar_use_num_now_ = 2;  // LiDAR 2만 데이터가 있으면 LiDAR 2 사용
        }
    }
    if (lidar_use_num_now_ == 0)  // LiDAR 번호가 0이면 에러
    {
        ROS_WARN("lidar_use_num_now_ is 0, error!");
        return false;
    }
    else if (lidar_use_num_now_ == 1)  // LiDAR 1 사용
    {
        /*** LiDAR 스캔 추가 ***/
        if(!lidar_pushed)  // 아직 LiDAR 데이터가 푸시되지 않은 경우
        {
            meas.lidar = lidar_buffer.front();  // 첫 번째 LiDAR 데이터를 측정 그룹에 추가
            meas.lidar_beg_time = time_buffer.front();  // LiDAR 시작 시간
            if (async_debug) cout << "\033[31;1mFirst LiDAR!, " << meas.lidar->points.size() << "\033[0m" << endl;
            current_lidar_num = 1;  // 현재 LiDAR 번호를 1로 설정

            if (meas.lidar->points.size() <= 1)  // 입력 포인트가 너무 적으면
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;  // LiDAR 종료 시간 설정
                ROS_WARN("Too few input point cloud!\n");
            }
            else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;  // 종료 시간 설정
            }
            else
            {
                scan_num ++;  // 스캔 수 증가
                lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);  // LiDAR 종료 시간 계산
                lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;  // 평균 스캔 시간 업데이트
            }

            meas.lidar_end_time = lidar_end_time;  // LiDAR 종료 시간 저장
            publish_lidar_time = lidar_end_time;  // 퍼블리시할 LiDAR 시간 업데이트
            lidar_pushed = true;  // LiDAR 푸시 완료
        }

        if (last_timestamp_imu < lidar_end_time)  // IMU 타임스탬프가 LiDAR 종료 시간보다 작으면
        {
            return false;  // 데이터 동기화 실패
        }

        /*** IMU 데이터를 푸시하고 IMU 버퍼에서 제거 ***/
        double imu_time = imu_buffer.front()->header.stamp.toSec();  // IMU 타임스탬프 가져오기
        meas.imu.clear();  // 측정 그룹의 IMU 데이터를 초기화
        while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))  // IMU 버퍼가 비어 있지 않고 IMU 타임스탬프가 LiDAR 종료 시간보다 작으면
        {
            imu_time = imu_buffer.front()->header.stamp.toSec();  // IMU 타임스탬프 가져오기
            if(imu_time > lidar_end_time) break;  // IMU 타임스탬프가 LiDAR 종료 시간을 초과하면 중단
            meas.imu.push_back(imu_buffer.front());  // 측정 그룹에 IMU 데이터 추가
            imu_buffer.pop_front();  // IMU 버퍼에서 제거
        }

        lidar_buffer.pop_front();  // LiDAR 버퍼에서 첫 번째 데이터 제거
        time_buffer.pop_front();  // 시간 버퍼에서 첫 번째 데이터 제거
        lidar_pushed = false;  // LiDAR 푸시 상태 초기화
        return true;  // 데이터 동기화 성공
    }
    else if (lidar_use_num_now_ == 2)  // LiDAR 2 사용
    {
        /*** LiDAR 스캔 추가 ***/
        if(!lidar_pushed)  // 아직 LiDAR 데이터가 푸시되지 않은 경우
        {
            meas.lidar2 = lidar_buffer2.front();  // 두 번째 LiDAR 데이터를 측정 그룹에 추가
            meas.lidar_beg_time2 = time_buffer2.front();  // LiDAR 시작 시간
            pcl::transformPointCloud(*meas.lidar2, *meas.lidar2, LiDAR2_wrt_LiDAR1);  // LiDAR 2 데이터를 LiDAR 1 좌표계로 변환
            if (async_debug) cout << "\033[32;1mSecond LiDAR!, " << meas.lidar2->points.size() << "\033[0m" << endl;
            current_lidar_num = 2;  // 현재 LiDAR 번호를 2로 설정

            if (meas.lidar2->points.size() <= 1)  // 입력 포인트가 너무 적으면
            {
                lidar_end_time2 = meas.lidar_beg_time2 + lidar_mean_scantime2;  // LiDAR 종료 시간 설정
                ROS_WARN("Too few input point cloud!\n");
            }
            else if (meas.lidar2->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime2)
            {
                lidar_end_time2 = meas.lidar_beg_time2 + lidar_mean_scantime2;  // 종료 시간 설정
            }
            else
            {
                scan_num2 ++;  // 스캔 수 증가
                lidar_end_time2 = meas.lidar_beg_time2 + meas.lidar2->points.back().curvature / double(1000);  // LiDAR 종료 시간 계산
                lidar_mean_scantime2 += (meas.lidar2->points.back().curvature / double(1000) - lidar_mean_scantime2) / scan_num2;  // 평균 스캔 시간 업데이트
            }

            meas.lidar_end_time2 = lidar_end_time2;  // LiDAR 종료 시간 저장
            publish_lidar_time = lidar_end_time2;  // 퍼블리시할 LiDAR 시간 업데이트
            lidar_pushed = true;  // LiDAR 푸시 완료
        }

        if (last_timestamp_imu < lidar_end_time2)  // IMU 타임스탬프가 LiDAR 종료 시간보다 작으면
        {
            return false;  // 데이터 동기화 실패
        }

        /*** IMU 데이터를 푸시하고 IMU 버퍼에서 제거 ***/
        double imu_time = imu_buffer.front()->header.stamp.toSec();  // IMU 타임스탬프 가져오기
        meas.imu.clear();  // 측정 그룹의 IMU 데이터를 초기화
        while ((!imu_buffer.empty()) && (imu_time < lidar_end_time2))  // IMU 버퍼가 비어 있지 않고 IMU 타임스탬프가 LiDAR 종료 시간보다 작으면
        {
            imu_time = imu_buffer.front()->header.stamp.toSec();  // IMU 타임스탬프 가져오기
            if(imu_time > lidar_end_time2) break;  // IMU 타임스탬프가 LiDAR 종료 시간을 초과하면 중단
            meas.imu.push_back(imu_buffer.front());  // 측정 그룹에 IMU 데이터 추가
            imu_buffer.pop_front();  // IMU 버퍼에서 제거
        }

        lidar_buffer2.pop_front();  // 두 번째 LiDAR 버퍼에서 첫 번째 데이터 제거
        time_buffer2.pop_front();  // 두 번째 시간 버퍼에서 첫 번째 데이터 제거
        lidar_pushed = false;  // LiDAR 푸시 상태 초기화
        return true;  // 데이터 동기화 성공
    }
}






bool sync_packages_bundle(MeasureGroup &meas)
{
    if (multi_lidar)
    {
        if (lidar_buffer.empty() || lidar_buffer2.empty() || imu_buffer.empty()) {
            return false;
        }
        /*** push a lidar scan ***/
        if(!lidar_pushed)
        {
            meas.lidar = lidar_buffer.front();
            for (size_t i = 1; i < lidar_buffer.size(); i++) // merge all lidar scans
            {
                *meas.lidar += *lidar_buffer[i];
            }
            meas.lidar_beg_time = time_buffer.front();
            if (meas.lidar->points.size() <= 1) // time too little
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
                ROS_WARN("Too few input point cloud!\n");
            }
            else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            }
            else
            {
                scan_num ++;
                lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
                lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
            }
            meas.lidar_end_time = lidar_end_time;

            meas.lidar2 = lidar_buffer2.front();
            for (size_t i = 1; i < lidar_buffer2.size(); i++) // merge all lidar scans
            {
                *meas.lidar2 += *lidar_buffer2[i];
            }
            pcl::transformPointCloud(*meas.lidar2, *meas.lidar2, LiDAR2_wrt_LiDAR1); //lidar2 data to lidar1 frame, not use lidar2-imu tf as state
            meas.lidar_beg_time2 = time_buffer2.front();
            if (meas.lidar2->points.size() <= 1) // time too little
            {
                lidar_end_time2 = meas.lidar_beg_time2 + lidar_mean_scantime2;
                ROS_WARN("Too few input point cloud!\n");
            }
            else if (meas.lidar2->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime2)
            {
                lidar_end_time2 = meas.lidar_beg_time2 + lidar_mean_scantime2;
            }
            else
            {
                scan_num2 ++;
                lidar_end_time2 = meas.lidar_beg_time2 + meas.lidar2->points.back().curvature / double(1000);
                lidar_mean_scantime2 += (meas.lidar2->points.back().curvature / double(1000) - lidar_mean_scantime2) / scan_num2;
            }
            meas.lidar_end_time2 = lidar_end_time2;

            publish_lidar_time = max(lidar_end_time, lidar_end_time2);
            lidar_pushed = true;
        }

        if (last_timestamp_imu < lidar_end_time || last_timestamp_imu < lidar_end_time2)
        {
            return false;
        }

        /*** push imu data, and pop from imu buffer ***/
        double imu_time = imu_buffer.front()->header.stamp.toSec();
        meas.imu.clear();
        while ((!imu_buffer.empty()) && (imu_time < lidar_end_time || imu_time < lidar_end_time2))
        {
            imu_time = imu_buffer.front()->header.stamp.toSec();
            if(imu_time > lidar_end_time && imu_time > lidar_end_time2) break;
            meas.imu.push_back(imu_buffer.front());
            imu_buffer.pop_front();
        }

        lidar_buffer.clear();
        time_buffer.clear();
        lidar_buffer2.clear();
        time_buffer2.clear();

        lidar_pushed = false;
        cout << "\033[36;1mBundle update!\033[0m" << endl;
        return true;
    }
    else
    {
        if (lidar_buffer.empty() || imu_buffer.empty()) {
            return false;
        }
        /*** push a lidar scan ***/
        if(!lidar_pushed)
        {
            meas.lidar = lidar_buffer.front();
            meas.lidar_beg_time = time_buffer.front();
            if (meas.lidar->points.size() <= 1) // time too little
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
                ROS_WARN("Too few input point cloud!\n");
            }
            else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            }
            else
            {
                scan_num ++;
                lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
                lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
            }

            meas.lidar_end_time = lidar_end_time;
            publish_lidar_time = lidar_end_time;
            lidar_pushed = true;
        }

        if (last_timestamp_imu < lidar_end_time)
        {
            return false;
        }

        /*** push imu data, and pop from imu buffer ***/
        double imu_time = imu_buffer.front()->header.stamp.toSec();
        meas.imu.clear();
        while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
        {
            imu_time = imu_buffer.front()->header.stamp.toSec();
            if(imu_time > lidar_end_time) break;
            meas.imu.push_back(imu_buffer.front());
            imu_buffer.pop_front();
        }

        lidar_buffer.pop_front();
        time_buffer.pop_front();

        lidar_pushed = false;
        return true;     
    }
}

void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty())
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_surf)*filter_size_surf + 0.5 * filter_size_surf;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_surf)*filter_size_surf + 0.5 * filter_size_surf;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_surf)*filter_size_surf + 0.5 * filter_size_surf;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_surf && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_surf && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_surf){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }
    ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    return;
}

void publish_frame_world(const ros::Publisher &pubLaserCloudFull, const ros::Publisher &pubLaserCloudFullTransFormed)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(publish_lidar_time);
        laserCloudmsg.header.frame_id = map_frame;
        pubLaserCloudFull.publish(laserCloudmsg);
        
        if (publish_tf_results)
        {
            PointCloudXYZI::Ptr laserCloudWorldTransFormed(new PointCloudXYZI(size, 1));
            pcl::transformPointCloud(*laserCloudWorld, *laserCloudWorldTransFormed, LiDAR1_wrt_drone);
            sensor_msgs::PointCloud2 laserCloudmsg2;
            pcl::toROSMsg(*laserCloudWorldTransFormed, laserCloudmsg2);
            laserCloudmsg2.header.stamp = ros::Time().fromSec(publish_lidar_time);
            laserCloudmsg2.header.frame_id = map_frame;
            pubLaserCloudFullTransFormed.publish(laserCloudmsg2);
        }
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

bool sync_packages_bundle(MeasureGroup &meas)
{
    if (multi_lidar)  // 다중 LiDAR 모드일 경우
    {
        if (lidar_buffer.empty() || lidar_buffer2.empty() || imu_buffer.empty()) {
            return false;  // LiDAR 또는 IMU 버퍼가 비어 있으면 false 반환
        }
        /*** LiDAR 스캔 병합 및 푸시 ***/
        if(!lidar_pushed)  // LiDAR 데이터가 아직 푸시되지 않은 경우
        {
            meas.lidar = lidar_buffer.front();  // 첫 번째 LiDAR 데이터를 측정 그룹에 추가
            for (size_t i = 1; i < lidar_buffer.size(); i++)  // 모든 LiDAR 스캔 병합
            {
                *meas.lidar += *lidar_buffer[i];  // LiDAR 데이터를 병합
            }
            meas.lidar_beg_time = time_buffer.front();  // LiDAR 시작 시간 설정
            if (meas.lidar->points.size() <= 1)  // 입력 포인트가 너무 적으면
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;  // LiDAR 종료 시간 설정
                ROS_WARN("Too few input point cloud!\n");
            }
            else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;  // LiDAR 종료 시간 설정
            }
            else
            {
                scan_num++;  // 스캔 수 증가
                lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);  // LiDAR 종료 시간 계산
                lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;  // 평균 스캔 시간 업데이트
            }
            meas.lidar_end_time = lidar_end_time;  // LiDAR 종료 시간 저장

            meas.lidar2 = lidar_buffer2.front();  // 두 번째 LiDAR 데이터를 측정 그룹에 추가
            for (size_t i = 1; i < lidar_buffer2.size(); i++)  // 모든 LiDAR 스캔 병합
            {
                *meas.lidar2 += *lidar_buffer2[i];  // LiDAR2 데이터를 병합
            }
            pcl::transformPointCloud(*meas.lidar2, *meas.lidar2, LiDAR2_wrt_LiDAR1);  // LiDAR2 데이터를 LiDAR1 좌표계로 변환
            meas.lidar_beg_time2 = time_buffer2.front();  // 두 번째 LiDAR 시작 시간 설정
            if (meas.lidar2->points.size() <= 1)  // 입력 포인트가 너무 적으면
            {
                lidar_end_time2 = meas.lidar_beg_time2 + lidar_mean_scantime2;  // LiDAR2 종료 시간 설정
                ROS_WARN("Too few input point cloud!\n");
            }
            else if (meas.lidar2->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime2)
            {
                lidar_end_time2 = meas.lidar_beg_time2 + lidar_mean_scantime2;  // LiDAR2 종료 시간 설정
            }
            else
            {
                scan_num2++;  // 두 번째 스캔 수 증가
                lidar_end_time2 = meas.lidar_beg_time2 + meas.lidar2->points.back().curvature / double(1000);  // LiDAR2 종료 시간 계산
                lidar_mean_scantime2 += (meas.lidar2->points.back().curvature / double(1000) - lidar_mean_scantime2) / scan_num2;  // 평균 스캔 시간 업데이트
            }
            meas.lidar_end_time2 = lidar_end_time2;  // 두 번째 LiDAR 종료 시간 저장

            publish_lidar_time = max(lidar_end_time, lidar_end_time2);  // 퍼블리시할 LiDAR 시간 설정
            lidar_pushed = true;  // LiDAR 데이터 푸시 완료
        }

        if (last_timestamp_imu < lidar_end_time || last_timestamp_imu < lidar_end_time2)  // IMU 타임스탬프가 LiDAR 종료 시간보다 작으면
        {
            return false;  // 데이터 동기화 실패
        }

        /*** IMU 데이터를 푸시하고 IMU 버퍼에서 제거 ***/
        double imu_time = imu_buffer.front()->header.stamp.toSec();  // 첫 번째 IMU 타임스탬프 가져오기
        meas.imu.clear();  // 측정 그룹의 IMU 데이터를 초기화
        while ((!imu_buffer.empty()) && (imu_time < lidar_end_time || imu_time < lidar_end_time2))  // IMU 데이터가 LiDAR 종료 시간보다 작으면
        {
            imu_time = imu_buffer.front()->header.stamp.toSec();  // IMU 타임스탬프 업데이트
            if (imu_time > lidar_end_time && imu_time > lidar_end_time2) break;  // IMU 타임스탬프가 LiDAR 종료 시간을 초과하면 중단
            meas.imu.push_back(imu_buffer.front());  // IMU 데이터를 측정 그룹에 추가
            imu_buffer.pop_front();  // IMU 버퍼에서 제거
        }

        lidar_buffer.clear();  // 첫 번째 LiDAR 버퍼 초기화
        time_buffer.clear();  // 첫 번째 시간 버퍼 초기화
        lidar_buffer2.clear();  // 두 번째 LiDAR 버퍼 초기화
        time_buffer2.clear();  // 두 번째 시간 버퍼 초기화

        lidar_pushed = false;  // LiDAR 데이터 푸시 상태 초기화
        cout << "\033[36;1mBundle update!\033[0m" << endl;  // 번들 업데이트 로그 출력
        return true;  // 데이터 동기화 성공
    }
    else  // 단일 LiDAR 모드일 경우
    {
        if (lidar_buffer.empty() || imu_buffer.empty()) {
            return false;  // LiDAR 또는 IMU 버퍼가 비어 있으면 false 반환
        }
        /*** LiDAR 스캔 푸시 ***/
        if(!lidar_pushed)  // LiDAR 데이터가 아직 푸시되지 않은 경우
        {
            meas.lidar = lidar_buffer.front();  // 첫 번째 LiDAR 데이터를 측정 그룹에 추가
            meas.lidar_beg_time = time_buffer.front();  // LiDAR 시작 시간 설정
            if (meas.lidar->points.size() <= 1)  // 입력 포인트가 너무 적으면
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;  // LiDAR 종료 시간 설정
                ROS_WARN("Too few input point cloud!\n");
            }
            else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;  // LiDAR 종료 시간 설정
            }
            else
            {
                scan_num++;  // 스캔 수 증가
                lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);  // LiDAR 종료 시간 계산
                lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;  // 평균 스캔 시간 업데이트
            }

            meas.lidar_end_time = lidar_end_time;  // LiDAR 종료 시간 저장
            publish_lidar_time = lidar_end_time;  // 퍼블리시할 LiDAR 시간 설정
            lidar_pushed = true;  // LiDAR 데이터 푸시 완료
        }

        if (last_timestamp_imu < lidar_end_time)  // IMU 타임스탬프가 LiDAR 종료 시간보다 작으면
        {
            return false;  // 데이터 동기화 실패
        }

        /*** IMU 데이터를 푸시하고 IMU 버퍼에서 제거 ***/
        double imu_time = imu_buffer.front()->header.stamp.toSec();  // 첫 번째 IMU 타임스탬프 가져오기
        meas.imu.clear();  // 측정 그룹의 IMU 데이터를 초기화
        while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))  // IMU 데이터가 LiDAR 종료 시간보다 작으면
        {
            imu_time = imu_buffer.front()->header.stamp.toSec();  // IMU 타임스탬프 업데이트
            if(imu_time > lidar_end_time) break;  // IMU 타임스탬프가 LiDAR 종료 시간을 초과하면 중단
            meas.imu.push_back(imu_buffer.front());  // IMU 데이터를 측정 그룹에 추가
            imu_buffer.pop_front();  // IMU 버퍼에서 제거
        }

        lidar_buffer.pop_front();  // LiDAR 버퍼에서 첫 번째 데이터 제거
        time_buffer.pop_front();  // 시간 버퍼에서 첫 번째 데이터 제거

        lidar_pushed = false;  // LiDAR 데이터 푸시 상태 초기화
        return true;  // 데이터 동기화 성공
    }
}

void map_incremental()
{
    PointVector PointToAdd;  // 추가할 포인트 벡터
    PointVector PointNoNeedDownsample;  // 다운샘플링이 필요 없는 포인트 벡터
    PointToAdd.reserve(feats_down_size);  // 추가할 포인트 벡터 크기 예약
    PointNoNeedDownsample.reserve(feats_down_size);  // 다운샘플링이 필요 없는 포인트 벡터 크기 예약
    for (int i = 0; i < feats_down_size; i++)
    {
        /* 월드 좌표계로 변환 */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* 맵에 추가할 필요가 있는지 결정 */
        if (!Nearest_Points[i].empty())  // 가까운 포인트가 있으면
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;  // 맵에 추가할지 여부
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x / filter_size_surf) * filter_size_surf + 0.5 * filter_size_surf;
            mid_point.y = floor(feats_down_world->points[i].y / filter_size_surf) * filter_size_surf + 0.5 * filter_size_surf;
            mid_point.z = floor(feats_down_world->points[i].z / filter_size_surf) * filter_size_surf + 0.5 * filter_size_surf;
            float dist  = calc_dist(feats_down_world->points[i], mid_point);  // 포인트와 미드포인트 간 거리 계산
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_surf && 
                fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_surf && 
                fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_surf) {
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);  // 다운샘플링이 필요 없으면 추가
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++)  // 일치하는 포인트 비교
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;  // 일치하는 포인트가 부족하면 중단
                if (calc_dist(points_near[readd_i], mid_point) < dist)  // 일치하는 포인트와의 거리가 더 짧으면
                {
                    need_add = false;  // 맵에 추가하지 않음
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);  // 맵에 추가해야 하면 포인트 추가
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);  // 가까운 포인트가 없으면 포인트 추가
        }
    }
    ikdtree.Add_Points(PointToAdd, true);  // KD-트리에 포인트 추가
    ikdtree.Add_Points(PointNoNeedDownsample, false);  // 다운샘플링이 필요 없는 포인트 추가
    return;
}

void publish_frame_world(const ros::Publisher &pubLaserCloudFull, const ros::Publisher &pubLaserCloudFullTransFormed)
{
    if(scan_pub_en)  // 스캔 퍼블리싱이 활성화된 경우
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);  // 밀집 퍼블리싱 여부에 따라 선택
        int size = laserCloudFullRes->points.size();  // 포인트 크기 가져오기
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));  // 월드 좌표계 포인트 클라우드 생성

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);  // 포인트를 월드 좌표계로 변환
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);  // 포인트 클라우드를 ROS 메시지로 변환
        laserCloudmsg.header.stamp = ros::Time().fromSec(publish_lidar_time);  // 타임스탬프 설정
        laserCloudmsg.header.frame_id = map_frame;  // 프레임 ID 설정
        pubLaserCloudFull.publish(laserCloudmsg);  // 포인트 클라우드 퍼블리시
        
        if (publish_tf_results)  // TF 결과 퍼블리싱이 활성화된 경우
        {
            PointCloudXYZI::Ptr laserCloudWorldTransFormed(new PointCloudXYZI(size, 1));  // 변환된 월드 포인트 클라우드 생성
            pcl::transformPointCloud(*laserCloudWorld, *laserCloudWorldTransFormed, LiDAR1_wrt_drone);  // LiDAR 1을 드론 좌표계로 변환
            sensor_msgs::PointCloud2 laserCloudmsg2;
            pcl::toROSMsg(*laserCloudWorldTransFormed, laserCloudmsg2);  // 포인트 클라우드를 ROS 메시지로 변환
            laserCloudmsg2.header.stamp = ros::Time().fromSec(publish_lidar_time);  // 타임스탬프 설정
            laserCloudmsg2.header.frame_id = map_frame;  // 프레임 ID 설정
            pubLaserCloudFullTransFormed.publish(laserCloudmsg2);  // 변환된 포인트 클라우드 퍼블리시
        }
    }

    /**************** 맵 저장 ****************/
    /* 1. 충분한 메모리 확보 필요
    /* 2. PCD 저장이 실시간 성능에 영향을 줄 수 있음 **/
    if (pcd_save_en)  // PCD 저장이 활성화된 경우
    {
        int size = feats_undistort->points.size();  // 포인트 크기 가져오기
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));  // 월드 포인트 클라우드 생성

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], &laserCloudWorld->points[i]);  // 포인트를 월드 좌표계로 변환
        }
        *pcl_wait_save += *laserCloudWorld;  // 저장 대기 중인 포인트 클라우드에 추가

        static int scan_wait_num = 0;  // 스캔 대기 번호
        scan_wait_num++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)  // PCD 저장 조건 확인
        {
            pcd_index++;  // PCD 인덱스 증가
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));  // PCD 파일 경로 설정
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;  // PCD 저장 로그 출력
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);  // PCD 파일 저장
            pcl_wait_save->clear();  // 저장 대기 클라우드 초기화
            scan_wait_num = 0;  // 스캔 대기 번호 초기화
        }
    }
}


void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();  // 왜곡을 보정한 포인트 수 가져오기
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));  // IMU 좌표계로 변환된 포인트 클라우드 생성

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], &laserCloudIMUBody->points[i]);  // LiDAR 포인트를 IMU 좌표계로 변환
    }

    sensor_msgs::PointCloud2 laserCloudmsg;  // ROS 메시지로 변환할 포인트 클라우드 메시지 생성
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);  // 포인트 클라우드를 ROS 메시지로 변환
    laserCloudmsg.header.stamp = ros::Time().fromSec(publish_lidar_time);  // 타임스탬프 설정
    laserCloudmsg.header.frame_id = "body";  // 프레임 ID를 'body'로 설정
    pubLaserCloudFull_body.publish(laserCloudmsg);  // 변환된 포인트 클라우드를 퍼블리시
}

void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;  // 맵을 위한 포인트 클라우드 메시지 생성
    pcl::toROSMsg(*featsFromMap, laserCloudMap);  // 포인트 클라우드를 ROS 메시지로 변환
    laserCloudMap.header.stamp = ros::Time().fromSec(publish_lidar_time);  // 타임스탬프 설정
    laserCloudMap.header.frame_id = map_frame;  // 프레임 ID를 'map'으로 설정
    pubLaserCloudMap.publish(laserCloudMap);  // 맵 포인트 클라우드를 퍼블리시
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);  // x 위치 설정
    out.pose.position.y = state_point.pos(1);  // y 위치 설정
    out.pose.position.z = state_point.pos(2);  // z 위치 설정
    out.pose.orientation.x = geoQuat.x;  // 쿼터니언 x 설정
    out.pose.orientation.y = geoQuat.y;  // 쿼터니언 y 설정
    out.pose.orientation.z = geoQuat.z;  // 쿼터니언 z 설정
    out.pose.orientation.w = geoQuat.w;  // 쿼터니언 w 설정
    return;
}


void publish_visionpose(const ros::Publisher &publisher)
{
    geometry_msgs::PoseStamped msg_out_;  // 퍼블리시할 메시지 생성
    msg_out_.header.frame_id = map_frame;  // 프레임 ID를 맵 프레임으로 설정
    msg_out_.header.stamp = ros::Time().fromSec(publish_lidar_time);  // 타임스탬프 설정

    Eigen::Matrix4d current_pose_eig_ = Eigen::Matrix4d::Identity();  // 현재 포즈 행렬 초기화
    current_pose_eig_.block<3, 3>(0, 0) = state_point.rot.toRotationMatrix();  // 회전 행렬 설정
    current_pose_eig_.block<3, 1>(0, 3) = state_point.pos;  // 위치 설정
    Eigen::Matrix4d tfed_vision_pose_eig_ = LiDAR1_wrt_drone * current_pose_eig_ * LiDAR1_wrt_drone.inverse();  
    // LiDAR와 드론 좌표계를 고려한 변환된 포즈 계산
    msg_out_.pose.position.x = tfed_vision_pose_eig_(0, 3);  // 변환된 x 위치 설정
    msg_out_.pose.position.y = tfed_vision_pose_eig_(1, 3);  // 변환된 y 위치 설정
    msg_out_.pose.position.z = tfed_vision_pose_eig_(2, 3);  // 변환된 z 위치 설정
    Eigen::Quaterniond tfed_quat_(tfed_vision_pose_eig_.block<3, 3>(0, 0));  // 회전을 쿼터니언으로 변환
    msg_out_.pose.orientation.x = tfed_quat_.x();  // 쿼터니언 x 설정
    msg_out_.pose.orientation.y = tfed_quat_.y();  // 쿼터니언 y 설정
    msg_out_.pose.orientation.z = tfed_quat_.z();  // 쿼터니언 z 설정
    msg_out_.pose.orientation.w = tfed_quat_.w();  // 쿼터니언 w 설정
    publisher.publish(msg_out_);  // 메시지 퍼블리시
    return;
}

void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = map_frame;  // 프레임 ID를 맵으로 설정
    odomAftMapped.child_frame_id = "body";  // 자식 프레임 ID를 바디로 설정
    odomAftMapped.header.stamp = ros::Time().fromSec(publish_lidar_time);  // 타임스탬프 설정
    set_posestamp(odomAftMapped.pose);  // 포즈 정보 설정
    pubOdomAftMapped.publish(odomAftMapped);  // 오도메트리 퍼블리시
    auto P = kf.get_P();  // 상태 추정 공분산 행렬 가져오기
    for (int i = 0; i < 6; i++)
    {
        int k = i < 3 ? i + 3 : i - 3;  // 공분산 인덱스 설정
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);  // 공분산 설정
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br_odom_to_body;  // TF 브로드캐스터 생성
    tf::Transform transform;  // 변환 정보 설정
    tf::Quaternion q;  // 쿼터니언 설정
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, 
                                    odomAftMapped.pose.pose.position.y, 
                                    odomAftMapped.pose.pose.position.z));  // 위치 설정
    q.setW(odomAftMapped.pose.pose.orientation.w);  // 쿼터니언 w 설정
    q.setX(odomAftMapped.pose.pose.orientation.x);  // 쿼터니언 x 설정
    q.setY(odomAftMapped.pose.pose.orientation.y);  // 쿼터니언 y 설정
    q.setZ(odomAftMapped.pose.pose.orientation.z);  // 쿼터니언 z 설정
    transform.setRotation(q);  // 회전 설정
    br_odom_to_body.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, map_frame, "body"));  
    // 오도메트리에서 바디로 변환 정보를 퍼블리시
}

void publish_path(const ros::Publisher pubPath)
{
    geometry_msgs::PoseStamped msg_body_pose;  // 바디 포즈 메시지 생성
    set_posestamp(msg_body_pose);  // 포즈 정보 설정
    msg_body_pose.header.stamp = ros::Time().fromSec(publish_lidar_time);  // 타임스탬프 설정
    msg_body_pose.header.frame_id = map_frame;  // 프레임 ID를 맵으로 설정

    /*** 경로가 너무 커지면 RViz가 충돌할 수 있음 ***/
    static int jjj = 0;  // 경로 퍼블리시 간격 설정
    jjj++;
    if (jjj % 10 == 0)  // 매 10번째 스캔마다 경로 업데이트
    {
        path.poses.push_back(msg_body_pose);  // 경로에 포즈 추가
        pubPath.publish(path);  // 경로 퍼블리시
    }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    laserCloudOri->clear();  // 원본 포인트 클라우드 초기화
    corr_normvect->clear();  // 정규 벡터 포인트 클라우드 초기화
    total_residual = 0.0;  // 총 잔차 초기화

    /** 근접한 표면을 찾고 잔차 계산 **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);  // OpenMP를 사용한 병렬 처리를 위해 스레드 수 설정
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)  // 다운샘플링된 모든 포인트에 대해 반복
    {
        PointType &point_body  = feats_down_body->points[i];  // 바디 좌표계에서 포인트 가져오기
        PointType &point_world = feats_down_world->points[i];  // 월드 좌표계에서 포인트 가져오기

        /* 월드 좌표계로 변환 */
        V3D p_body(point_body.x, point_body.y, point_body.z);  // 바디 좌표계에서 포인트
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);  // 월드 좌표계로 변환된 포인트
        point_world.x = p_global(0);  // 변환된 x 좌표
        point_world.y = p_global(1);  // 변환된 y 좌표
        point_world.z = p_global(2);  // 변환된 z 좌표
        point_world.intensity = point_body.intensity;  // 강도 값 복사

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);  // 근접 포인트들의 거리 저장
        auto &points_near = Nearest_Points[i];  // 근접 포인트들 저장

        if (ekfom_data.converge)  // EKF 데이터가 수렴한 경우
        {
            /** 맵에서 가장 가까운 표면 찾기 **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);  // KD-트리 검색
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;  
            // 근접 포인트 수가 충분하지 않거나 거리가 너무 멀면 표면 선택 안 함
        }

        if (!point_selected_surf[i]) continue;  // 선택된 표면이 없으면 다음 포인트로 이동

        VF(4) pabcd;  // 평면 방정식 저장
        point_selected_surf[i] = false;  // 표면 선택 여부 초기화
        if (esti_plane(pabcd, points_near, 0.1f))  // 평면 추정 성공 시
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);  // 포인트와 평면의 거리 계산
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());  // 거리 기반 가중치 계산

            if (s > 0.9)  // 가중치가 0.9 이상이면
            {
                point_selected_surf[i] = true;  // 표면 선택
                normvec->points[i].x = pabcd(0);  // 법선 벡터 설정
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;  // 잔차 설정
                res_last[i] = abs(pd2);  // 잔차 절댓값 저장
            }
        }
    }
    
    effect_feat_num = 0;  // 유효한 특징점 수 초기화
    localizability_vec = Eigen::Vector3d::Zero();  // 지역화 가능성 벡터 초기화
    for (int i = 0; i < feats_down_size; i++)  // 다운샘플링된 포인트를 순회
    {
        if (point_selected_surf[i])  // 선택된 표면에 대해
        {
            laserCloudOri->points[effect_feat_num] = feats_down_body->points[i];  // 바디 포인트 저장
            corr_normvect->points[effect_feat_num] = normvec->points[i];  // 정규 벡터 저장
            total_residual += res_last[i];  // 총 잔차 계산
            effect_feat_num++;  // 유효한 특징점 수 증가
            localizability_vec += Eigen::Vector3d(normvec->points[i].x, normvec->points[i].y, normvec->points[i].z).array().square().matrix();  // 지역화 가능성 계산
        }
    }
    localizability_vec = localizability_vec.cwiseSqrt();  // 지역화 가능성의 제곱근 계산

    if (effect_feat_num < 1)  // 유효한 특징점이 없으면
    {
        ekfom_data.valid = false;  // EKF 데이터 무효화
        ROS_WARN("No Effective Points! \n");  // 경고 메시지 출력
        return;
    }

    res_mean_last = total_residual / effect_feat_num;  // 평균 잔차 계산
        
    /*** 측정 자코비안 행렬 H와 측정 벡터 계산 ***/
    ekfom_data.h_x = MatrixXd::Zero(effect_feat_num, 12);  // 측정 자코비안 행렬 초기화
    ekfom_data.h.resize(effect_feat_num);  // 측정 벡터 크기 설정

    for (int i = 0; i < effect_feat_num; i++)  // 유효한 특징점에 대해
    {
        const PointType &laser_p  = laserCloudOri->points[i];  // 바디 좌표계에서의 포인트
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);  // 바디 좌표계 포인트 벡터
        M3D point_be_crossmat;  // 바디 포인트의 교차곱 행렬
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);  // 반대칭 행렬 계산
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;  // LiDAR에서 IMU로 변환된 포인트
        M3D point_crossmat;  // 변환된 포인트의 교차곱 행렬
        point_crossmat<<SKEW_SYM_MATRX(point_this);  // 반대칭 행렬 계산

        /*** 근접 표면/코너의 법선 벡터 가져오기 ***/
        const PointType &norm_p = corr_normvect->points[i];  // 법선 벡터
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);  // 법선 벡터로 변환

        /*** 측정 자코비안 행렬 H 계산 ***/
        V3D C(s.rot.conjugate() *norm_vec);  // 회전 행렬을 고려한 법선 벡터
        V3D A(point_crossmat * C);  // 자코비안 계산
        if (extrinsic_est_en)  // 외부 요소 추정이 활성화된 경우
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C);  // 자코비안 계산
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);  // 자코비안 행렬에 값 설정
        }
        else  // 외부 요소 추정이 비활성화된 경우
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;  // 자코비안 행렬에 값 설정
        }

        /*** 측정: 가장 가까운 표면/코너까지의 거리 ***/
        ekfom_data.h(i) = -norm_p.intensity;  // 거리 값을 측정 벡터에 설정
    }
}

/**** 
localizability는 로봇이 주변 환경에서 자신의 위치를 얼마나 정확하게 추정할 수 있는지를 나타내는 지표입니다. 
이는 로봇이 인식한 표면의 법선 벡터(normvec)를 기반으로 계산됩니다.

1. **법선 벡터의 역할**
   법선 벡터는 표면의 방향을 나타냅니다. 로봇이 주변 환경의 표면 방향을 인식하면, 
   자신의 위치를 추정하는 데 필요한 정보를 얻을 수 있습니다. 
   법선 벡터가 잘 정의되어 있다는 것은 로봇이 그 환경에서 유용한 정보를 얻고 있다는 뜻입니다.

2. **localizability 계산**
   코드에서는 각 포인트의 법선 벡터를 제곱하여 `localizability_vec`에 누적합니다:
   
   //localizability_vec += Eigen::Vector3d(normvec->points[i].x, normvec->points[i].y, normvec->points[i].z).array().square().matrix();
   
   이 계산은 x, y, z 축에서 얻은 정보를 평가하는 과정입니다. 
   특정 축에서 법선 벡터의 값이 크다면, 그 축에서 많은 정보를 얻고 있다는 의미입니다.

3. **localizability의 중요성**
   로봇이 환경에서 정확하게 위치를 추정하려면 다양한 방향에서 충분한 정보를 감지해야 합니다. 
   예를 들어, 평평한 벽만 감지한다면 로봇은 벽에 대해 자신의 위치를 파악하기 어려울 수 있습니다. 
   반면, 다양한 물체와 지형이 있는 환경에서는 법선 벡터가 다양한 방향에서 분포되어 
   로봇이 더 많은 정보를 기반으로 위치를 추정할 수 있게 됩니다.

4. **localizability가 높을 때**
   법선 벡터가 다양한 방향에서 충분한 정보를 제공하면, 그 환경은 로봇이 자신의 위치를 잘 추정할 수 있는 환경입니다. 
   즉, **localizability**가 높으면 로봇이 해당 환경에서 위치를 추정하기에 유리한 환경이라는 뜻입니다.

5. **결론**
   localizability는 로봇이 주변 환경에서 얻는 정보의 양과 방향성을 평가하는 지표입니다. 
   SLAM(Simultaneous Localization and Mapping)과 같은 시스템에서 매우 중요한 역할을 하며, 
   로봇이 자신의 위치를 얼마나 정확하게 추정할 수 있는지를 나타냅니다.

---

이제 localizability 개념을 SLAM 성능 최적화에 어떻게 적용할 수 있는지 설명하겠습니다.

6. **Localizability가 SLAM 최적화에 유리한 이유**
   SLAM에서 로봇의 위치를 추정할 때, 로봇이 주변 환경에서 얻는 정보의 양과 방향성은 매우 중요합니다. 
   localizability는 이러한 정보를 수치화하여 특정 환경에서 로봇이 얼마나 정확하게 위치를 추정할 수 있는지를 평가합니다.
   
   - **정보가 풍부한 환경**: 다양한 구조물과 물체가 있는 복잡한 환경에서는 법선 벡터가 다양한 방향에서 제공되며, 이는 높은 localizability를 의미합니다. 이 경우, SLAM 알고리즘은 더 많은 유용한 정보를 바탕으로 정확한 맵을 생성할 수 있습니다.
   - **정보가 부족한 환경**: 반대로, 평평한 벽이나 단순한 구조가 있는 공간에서는 법선 벡터가 단일 방향으로만 분포해 localizability가 낮아집니다. 이러한 환경에서는 SLAM 성능이 저하될 수 있으며, 위치 추정에 어려움을 겪을 수 있습니다.

7. **XICP 논문에서의 활용**
   XICP(Extended ICP)는 localizability 개념을 포함한 확장된 ICP 알고리즘으로, SLAM 성능을 최적화하는 방법을 제안합니다. XICP는 기존의 ICP(Iterative Closest Point) 알고리즘을 개선하여 포인트 클라우드 매칭에서 더 나은 성능을 제공합니다. 여기에 localizability 개념을 적용하면 다음과 같은 최적화가 가능합니다:

   - **환경 특성에 따른 적응형 최적화**: localizability 값에 따라 SLAM 알고리즘이 환경에 적응해 최적의 매칭 방법을 선택합니다. 복잡한 환경에서는 더 많은 정보를 활용하고, 단순한 환경에서는 덜 신뢰할 수 있는 정보를 필터링하는 방식으로 성능을 향상시킬 수 있습니다.
   - **정보 가중치 부여**: localizability 값에 따라 포인트 클라우드 데이터에 가중치를 부여해, 더 유용한 정보에 높은 신뢰도를 할당할 수 있습니다. 이를 통해 덜 유용한 정보는 최소화하고, SLAM 알고리즘이 더 정확한 결과를 낼 수 있습니다.

8. **다른 논문에서의 응용**
   localizability를 활용한 다양한 SLAM 최적화 방법은 다음과 같습니다:
   - **정보 이득 기반 탐색**: 로봇이 환경을 탐색할 때 localizability가 높은 지역을 우선적으로 탐색하여 정보 이득을 극대화하는 방식.
   - **경로 계획 최적화**: localizability를 고려한 경로 계획을 통해, 로봇이 위치 추정에 필요한 정보를 충분히 얻을 수 있는 경로를 우선적으로 탐색함으로써 성능을 최적화합니다.
   - **위치 정확도 평가**: localizability를 기반으로 로봇이 위치 추정이 어려운 환경을 인식하고, 경로를 수정하거나 보정하여 안정적인 위치 추정을 할 수 있습니다.

9. **결론**
   localizability를 SLAM 성능 최적화에 적용하는 것은 매우 유용한 전략입니다. SLAM에서 핵심은 로봇이 환경에서 자신의 위치를 얼마나 정확하게 추정하고 이를 통해 맵을 생성할 수 있는가에 있습니다. localizability는 로봇이 주변에서 얻는 정보의 양과 질을 수치화하는 도구로, 이를 통해 SLAM 알고리즘이 더욱 효율적으로 작동할 수 있습니다.
   
   특히 자율 주행 차량이나 로봇 탐사와 같은 응용 분야에서, localizability를 고려한 SLAM 최적화는 시스템의 안정성과 성능을 크게 향상시킬 수 있습니다.
****/

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");  // ROS 노드 초기화, 노드 이름은 "laserMapping"
    ros::NodeHandle nh;  // ROS 노드 핸들 생성

    // temporal variables (외부 파라미터 저장을 위한 임시 변수들)
    vector<double> extrinT(3, 0.0);  // 첫 번째 LiDAR의 외부 변환 행렬 (변위)
    vector<double> extrinR(9, 0.0);  // 첫 번째 LiDAR의 외부 회전 행렬
    vector<double> extrinT2(3, 0.0);  // 두 번째 LiDAR의 외부 변환 행렬 (변위)
    vector<double> extrinR2(9, 0.0);  // 두 번째 LiDAR의 외부 회전 행렬
    vector<double> extrinT3(3, 0.0);  // LiDAR2가 LiDAR1에 대해 변환된 외부 변위
    vector<double> extrinR3(9, 0.0);  // LiDAR2가 LiDAR1에 대해 변환된 외부 회전
    vector<double> extrinT4(3, 0.0);  // LiDAR1이 드론에 대해 변환된 외부 변위
    vector<double> extrinR4(9, 0.0);  // LiDAR1이 드론에 대해 변환된 외부 회전

    // ROS 파라미터 서버에서 설정값을 불러옴
    nh.param<int>("common/max_iteration", NUM_MAX_ITERATIONS, 4);  // 최대 반복 횟수 설정
    nh.param<bool>("common/async_debug", async_debug, false);  // 비동기 디버깅 모드 여부 설정
    nh.param<bool>("common/multi_lidar", multi_lidar, true);  // 멀티 LiDAR 사용 여부 설정
    nh.param<bool>("common/publish_tf_results", publish_tf_results, true);  // TF 결과를 퍼블리시할지 여부
    nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");  // 첫 번째 LiDAR 데이터 토픽 이름
    nh.param<string>("common/lid_topic2", lid_topic2, "/livox/lidar");  // 두 번째 LiDAR 데이터 토픽 이름
    nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");  // IMU 데이터 토픽 이름
    nh.param<string>("common/map_frame", map_frame, "map");  // 맵 프레임 이름 설정

    // SLAM 성능 관련 매개변수
    nh.param<int>("method/voxelized_pt_num_thres", voxelized_pt_num_thres, 100);  // voxel 그리드에 포함될 최소 포인트 수
    nh.param<double>("method/effect_pt_num_ratio_thres", effect_pt_num_ratio_thres, 0.4);  // 유효 포인트 수 비율 임계값
    nh.param<int>("method/bundle_enabled_tic_thres", bundle_enabled_tic_thres, 5);  // 번들 업데이트가 활성화되는 임계값

    // LiDAR 전처리 관련 파라미터 설정
    nh.param<double>("preprocess/filter_size_surf", filter_size_surf, 0.5);  // 필터 사이즈 설정 (표면)
    nh.param<int>("preprocess/point_filter_num", p_pre->point_filter_num[0], 2);  // 첫 번째 LiDAR 필터 포인트 수
    nh.param<int>("preprocess/point_filter_num2", p_pre->point_filter_num[1], 2);  // 두 번째 LiDAR 필터 포인트 수
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type[0], AVIA);  // 첫 번째 LiDAR 타입 설정
    nh.param<int>("preprocess/lidar_type2", p_pre->lidar_type[1], AVIA);  // 두 번째 LiDAR 타입 설정
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS[0], 16);  // 첫 번째 LiDAR 스캔 라인 수
    nh.param<int>("preprocess/scan_line2", p_pre->N_SCANS[1], 16);  // 두 번째 LiDAR 스캔 라인 수
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE[0], 10);  // 첫 번째 LiDAR 스캔 속도
    nh.param<int>("preprocess/scan_rate2", p_pre->SCAN_RATE[1], 10);  // 두 번째 LiDAR 스캔 속도
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit[0], US);  // 첫 번째 LiDAR 타임스탬프 단위
    nh.param<int>("preprocess/timestamp_unit2", p_pre->time_unit[1], US);  // 두 번째 LiDAR 타임스탬프 단위
    nh.param<double>("preprocess/blind", p_pre->blind[0], 0.01);  // 첫 번째 LiDAR의 블라인드 영역 설정
    nh.param<double>("preprocess/blind2", p_pre->blind[1], 0.01);  // 두 번째 LiDAR의 블라인드 영역 설정
    p_pre->set();  // 전처리 설정 적용

    // 매핑 관련 파라미터 설정
    nh.param<double>("mapping/cube_side_length", cube_len, 200.0);  // 매핑 큐브의 한 변 길이
    nh.param<float>("mapping/det_range", DET_RANGE, 300.f);  // 매핑 감지 범위
    nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);  // 자이로스코프 공분산 설정
    nh.param<double>("mapping/acc_cov", acc_cov, 0.1);  // 가속도계 공분산 설정
    nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);  // 자이로 바이어스 공분산
    nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);  // 가속도계 바이어스 공분산
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);  // 외부 파라미터 추정 활성화 여부
    nh.param<bool>("mapping/extrinsic_imu_to_lidars", extrinsic_imu_to_lidars, true);  // IMU와 LiDAR 간 외부 파라미터 추정 여부
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());  // 첫 번째 LiDAR 외부 변환 파라미터 (변위)
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());  // 첫 번째 LiDAR 외부 회전 파라미터
    nh.param<vector<double>>("mapping/extrinsic_T2", extrinT2, vector<double>());  // 두 번째 LiDAR 외부 변환 파라미터 (변위)
    nh.param<vector<double>>("mapping/extrinsic_R2", extrinR2, vector<double>());  // 두 번째 LiDAR 외부 회전 파라미터
    nh.param<vector<double>>("mapping/extrinsic_T_L2_wrt_L1", extrinT3, vector<double>());  // LiDAR2가 LiDAR1에 대해 변환되는 외부 파라미터 (변위)
    nh.param<vector<double>>("mapping/extrinsic_R_L2_wrt_L1", extrinR3, vector<double>());  // LiDAR2가 LiDAR1에 대해 변환되는 외부 파라미터 (회전)
    nh.param<vector<double>>("mapping/extrinsic_T_L1_wrt_drone", extrinT4, vector<double>());  // LiDAR1이 드론에 대해 변환되는 외부 파라미터 (변위)
    nh.param<vector<double>>("mapping/extrinsic_R_L1_wrt_drone", extrinR4, vector<double>());  // LiDAR1이 드론에 대해 변환되는 외부 파라미터 (회전)

    // 퍼블리싱 관련 파라미터 설정
    nh.param<bool>("publish/path_en", path_en, true);  // 경로 퍼블리싱 활성화 여부
    nh.param<bool>("publish/scan_publish_en", scan_pub_en, true);  // 스캔 퍼블리싱 활성화 여부
    nh.param<bool>("publish/dense_publish_en", dense_pub_en, true);  // 고밀도 스캔 퍼블리싱 여부
    nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en, true);  // 본체 프레임에서의 스캔 퍼블리싱 여부
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);  // PCD 저장 활성화 여부
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);  // PCD 저장 간격 설정




    /*** 변수 초기화 ***/
    path.header.stamp = ros::Time::now();  // 현재 시간을 경로 메시지의 헤더에 설정
    path.header.frame_id = map_frame;  // 경로 메시지의 프레임 ID를 맵 프레임으로 설정
    memset(point_selected_surf, true, sizeof(point_selected_surf));  // 포인트 선택 여부를 모두 true로 초기화
    memset(res_last, -1000.0f, sizeof(res_last));  // 이전 잔차(residual)를 -1000.0으로 초기화
    downSizeFilterSurf.setLeafSize(filter_size_surf, filter_size_surf, filter_size_surf);  // 다운샘플링 필터의 크기 설정
    ikdtree.set_downsample_param(filter_size_surf);  // kdtree의 다운샘플 파라미터 설정

    V3D Lidar_T_wrt_IMU(Zero3d);  // LiDAR와 IMU 사이의 변위 초기화
    M3D Lidar_R_wrt_IMU(Eye3d);  // LiDAR와 IMU 사이의 회전 행렬 초기화
    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);  // 외부 파라미터로부터 변위 값을 설정
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);  // 외부 파라미터로부터 회전 값을 설정
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);  // IMU의 외부 파라미터 설정
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));  // 자이로스코프 공분산 설정
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));  // 가속도계 공분산 설정
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));  // 자이로 바이어스 공분산 설정
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));  // 가속도계 바이어스 공분산 설정

    // 멀티 LiDAR의 경우
    if (multi_lidar)
    {
        if (extrinsic_imu_to_lidars)  // IMU와 LiDAR 사이의 외부 파라미터 사용 여부
        {
            Eigen::Matrix4d Lidar_wrt_IMU = Eigen::Matrix4d::Identity();  // LiDAR1과 IMU 간 변환 행렬 초기화
            Eigen::Matrix4d Lidar2_wrt_IMU = Eigen::Matrix4d::Identity();  // LiDAR2와 IMU 간 변환 행렬 초기화
            V3D LiDAR2_T_wrt_IMU; LiDAR2_T_wrt_IMU << VEC_FROM_ARRAY(extrinT2);  // LiDAR2와 IMU 간 변위 값 설정
            M3D LiDAR2_R_wrt_IMU; LiDAR2_R_wrt_IMU << MAT_FROM_ARRAY(extrinR2);  // LiDAR2와 IMU 간 회전 값 설정
            Lidar_wrt_IMU.block<3,3>(0,0) = Lidar_R_wrt_IMU;  // LiDAR1의 회전 값 설정
            Lidar_wrt_IMU.block<3,1>(0,3) = Lidar_T_wrt_IMU;  // LiDAR1의 변위 값 설정
            Lidar2_wrt_IMU.block<3,3>(0,0) = LiDAR2_R_wrt_IMU;  // LiDAR2의 회전 값 설정
            Lidar2_wrt_IMU.block<3,1>(0,3) = LiDAR2_T_wrt_IMU;  // LiDAR2의 변위 값 설정
            LiDAR2_wrt_LiDAR1 = Lidar_wrt_IMU.inverse() * Lidar2_wrt_IMU;  // LiDAR2가 LiDAR1에 대해 변환된 변환 행렬 계산
        }
        else
        {
            V3D LiDAR2_T_wrt_LiDAR1; LiDAR2_T_wrt_LiDAR1 << VEC_FROM_ARRAY(extrinT3);  // LiDAR2와 LiDAR1 간 변위 설정
            M3D Lidar2_R_wrt_LiDAR1; Lidar2_R_wrt_LiDAR1 << MAT_FROM_ARRAY(extrinR3);  // LiDAR2와 LiDAR1 간 회전 설정
            LiDAR2_wrt_LiDAR1.block<3,3>(0,0) = Lidar2_R_wrt_LiDAR1;  // 회전 값 설정
            LiDAR2_wrt_LiDAR1.block<3,1>(0,3) = LiDAR2_T_wrt_LiDAR1;  // 변위 값 설정
        }
        cout << "\033[32;1mMulti LiDAR on!" << endl;  // 멀티 LiDAR 활성화 알림 출력
        cout << "lidar_type[0]: " << p_pre->lidar_type[0] << ", " << "lidar_type[1]: " << p_pre->lidar_type[1] << endl << endl;  // LiDAR 타입 출력
        cout << "L2 wrt L1 TF: " << endl << LiDAR2_wrt_LiDAR1 << "\033[0m" << endl << endl;  // LiDAR2가 LiDAR1에 대해 변환된 변환 행렬 출력
    }
    if (publish_tf_results)  // TF 결과 퍼블리싱 여부 확인
    {
        V3D LiDAR1_T_wrt_drone; LiDAR1_T_wrt_drone << VEC_FROM_ARRAY(extrinT4);  // LiDAR1과 드론 간 변위 설정
        M3D LiDAR2_R_wrt_drone; LiDAR2_R_wrt_drone << MAT_FROM_ARRAY(extrinR4);  // LiDAR1과 드론 간 회전 설정
        LiDAR1_wrt_drone.block<3,3>(0,0) = LiDAR2_R_wrt_drone;  // 회전 값 설정
        LiDAR1_wrt_drone.block<3,1>(0,3) = LiDAR1_T_wrt_drone;  // 변위 값 설정
        cout << "\033[32;1mLiDAR wrt Drone:" << endl;  // LiDAR와 드론 간 변환 행렬 출력
        cout << LiDAR1_wrt_drone << "\033[0m" << endl << endl;
    }

    double epsi[23] = {0.001};  // 초기화된 공차 값 배열 설정
    fill(epsi, epsi+23, 0.001);  // 모든 공차 값을 0.001로 채움
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);  // Kalman 필터 초기화

    /*** ROS 구독자 초기화 ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type[0] == AVIA ? \
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 200000, standard_pcl_cbk);  // 첫 번째 LiDAR 데이터 구독 설정
    ros::Subscriber sub_pcl2;
    if (multi_lidar)  // 멀티 LiDAR가 활성화된 경우
    {
        sub_pcl2 = p_pre->lidar_type[1] == AVIA ? \
            nh.subscribe(lid_topic2, 200000, livox_pcl_cbk2) : \
            nh.subscribe(lid_topic2, 200000, standard_pcl_cbk2);  // 두 번째 LiDAR 데이터 구독 설정
    }


    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);  
    // IMU 데이터를 수신하는 구독자(subscriber) 설정. 
    // "/livox/imu" 토픽에서 IMU 데이터를 받아 콜백 함수 imu_cbk로 처리함.

    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered", 100000);
    // LiDAR로 수집된 포인트 클라우드 데이터를 퍼블리싱하는 퍼블리셔. 
    // "/cloud_registered" 토픽으로 PointCloud2 메시지 형태로 데이터를 퍼블리싱함.

    ros::Publisher pubLaserCloudFullTransformed = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_tf", 100000);
    // 변환된 포인트 클라우드를 퍼블리싱하는 퍼블리셔. 
    // "/cloud_registered_tf" 토픽에 PointCloud2 메시지로 퍼블리싱.

    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_body", 100000);
    // LiDAR로부터 얻은 포인트 클라우드를 로봇 몸체 좌표계 기준으로 퍼블리싱.
    // "/cloud_registered_body" 토픽에 퍼블리싱.

    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
            ("/Laser_map", 100000);
    // 맵에 해당하는 포인트 클라우드 데이터를 퍼블리싱하는 퍼블리셔. 
    // "/Laser_map" 토픽으로 맵 데이터를 퍼블리싱.

    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> 
            ("/Odometry", 100000);
    // SLAM 결과로 나온 로봇의 위치 및 속도를 퍼블리싱하는 퍼블리셔. 
    // "/Odometry" 토픽으로 Odometry 메시지 퍼블리싱.

    ros::Publisher pubMavrosVisionPose = nh.advertise<geometry_msgs::PoseStamped> 
            ("/mavros/vision_pose/pose", 100000);    
    // MAVROS 패키지를 사용하여 로봇의 비전 기반 자세 정보를 퍼블리싱.
    // "/mavros/vision_pose/pose" 토픽에 PoseStamped 메시지로 퍼블리싱.

    ros::Publisher pubPath = nh.advertise<nav_msgs::Path> 
            ("/path", 100000);
    // 로봇이 이동한 경로를 퍼블리싱하는 퍼블리셔. 
    // "/path" 토픽으로 경로를 퍼블리싱.

    ros::Publisher pubCaclTime = nh.advertise<std_msgs::Float32> 
            ("/calc_time", 100000);
    // 계산 시간을 퍼블리싱하는 퍼블리셔. 
    // "/calc_time" 토픽에 Float32 타입으로 계산 시간 퍼블리싱.

    ros::Publisher pubPointNum = nh.advertise<std_msgs::Float32> 
            ("/point_number", 100000);
    // 포인트 클라우드의 포인트 수를 퍼블리싱하는 퍼블리셔. 
    // "/point_number" 토픽에 Float32 타입으로 포인트 개수 퍼블리싱.

    ros::Publisher pubLocalizabilityX = nh.advertise<std_msgs::Float32> 
            ("/localizability_x", 100000);
    // X축에 대한 로컬라이저빌리티(위치 추정 가능성)를 퍼블리싱하는 퍼블리셔. 
    // "/localizability_x" 토픽에 Float32 메시지로 퍼블리싱.

    ros::Publisher pubLocalizabilityY = nh.advertise<std_msgs::Float32> 
            ("/localizability_y", 100000);
    // Y축에 대한 로컬라이저빌리티 퍼블리셔. 
    // "/localizability_y" 토픽에 퍼블리싱.

    ros::Publisher pubLocalizabilityZ = nh.advertise<std_msgs::Float32> 
            ("/localizability_z", 100000);
    // Z축에 대한 로컬라이저빌리티 퍼블리셔. 
    // "/localizability_z" 토픽에 퍼블리싱.

        // SIGINT 신호 핸들러를 설정하여 안전하게 프로그램을 종료할 수 있도록 함 (예: Ctrl+C)
    signal(SIGINT, SigHandle); 

    // 루프 속도를 5000Hz로 설정
    ros::Rate rate(5000);

    // ROS 실행 상태를 추적하는 플래그
    bool status = ros::ok();
    
    // 메인 처리 루프
    while (status)
    {
        if (flg_exit) break;  // 종료 플래그가 설정되면 루프 종료

        // 한 번의 ROS 메시지 처리
        ros::spinOnce();

        // 동기화 상태를 추적하는 플래그 초기화
        bool synced = false;

        // 번들 처리가 활성화된 경우, 번들 동기화 메소드 사용
        if (bundle_enabled) 
            synced = sync_packages_bundle(Measures);  // 번들 처리 (여러 센서 동기화)
        else 
            synced = sync_packages_async(Measures);   // 단일 또는 비동기 처리

        // 동기화가 성공적으로 이루어진 경우 처리 진행
        if(synced) 
        {
            // 처리 시간을 측정하기 위한 타이머 시작
            high_resolution_clock::time_point t1 = high_resolution_clock::now();

            // 번들 모드에 따라 IMU와 LiDAR 데이터를 처리
            if (bundle_enabled) 
                p_imu->Process(Measures, kf, feats_undistort, multi_lidar);  // 번들 처리
            else 
                p_imu->Process(Measures, kf, feats_undistort, false, current_lidar_num);  // 비동기/단일 처리

            // 칼만 필터에서 현재 상태(포즈와 회전)를 가져옴
            state_point = kf.get_x();

            // 현재 상태를 기반으로 LiDAR 위치 업데이트
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            // 처리할 특징 포인트가 없으면 스캔 건너뛰기
            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("포인트가 없습니다, 이 스캔을 건너뜁니다!\n");
                continue;  // 다음 루프 반복으로 건너뜀
            }

            /*** LiDAR의 FOV 내에서 맵 세그먼트 ***/
            lasermap_fov_segment();

            /*** 스캔에서 특징 포인트 다운샘플링 ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            feats_down_size = feats_down_body->points.size();  // 다운샘플링된 포인트 수 가져오기

            /*** 처음 스캔일 경우 맵 kdtree 초기화 ***/
            if (multi_lidar)
            {
                // kdtree가 초기화되지 않았거나 LiDAR 데이터가 준비되지 않은 경우 초기화
                if(ikdtree.Root_Node == nullptr || !lidar1_ikd_init || !lidar2_ikd_init)
                {
                    // kdtree를 초기화하기에 충분한 특징 포인트가 있는지 확인
                    if(feats_down_size > 5)
                    {
                        // 포인트를 바디 프레임에서 월드 프레임으로 변환
                        feats_down_world->resize(feats_down_size);
                        for (int i = 0; i < feats_down_size; i++)
                        {
                            pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                        }                    
                        // 월드 프레임 포인트를 kdtree에 추가
                        ikdtree.Add_Points(feats_down_world->points, true);
                        
                        // LiDAR를 초기화된 것으로 표시
                        if (current_lidar_num == 1)
                        {
                            lidar1_ikd_init = true;
                        }
                        else if (current_lidar_num == 2)
                        {
                            lidar2_ikd_init = true;
                        }
                    }
                    continue;  // 초기화 후 다음 루프 반복으로 건너뜀
                }
            }
            // 단일 LiDAR 시스템의 경우, 필요 시 kdtree 초기화
            else if(ikdtree.Root_Node == nullptr)
            {
                if(feats_down_size > 5)
                {
                    feats_down_world->resize(feats_down_size);
                    for (int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }                    
                    ikdtree.Add_Points(feats_down_world->points, true);  // 포인트를 kdtree에 추가
                }
                continue;  // 초기화 후 다음 루프 반복으로 건너뜀
            }


               /*** ICP 및 반복 칼만 필터 업데이트 ***/
    if (feats_down_size < 5)  // 다운샘플링된 포인트가 너무 적을 경우 스캔을 건너뜀
    {
        ROS_WARN("포인트가 없으므로 스캔을 건너뜁니다!\n");
        continue;  // 다음 루프로 넘어감
    }
    
    // 각 변수의 크기를 다운샘플된 포인트 수에 맞게 조정
    normvec->resize(feats_down_size);
    feats_down_world->resize(feats_down_size);
    Nearest_Points.resize(feats_down_size);

    /*** 반복적인 상태 추정 업데이트 ***/
    double solve_H_time = 0;  // H 행렬을 푸는 데 걸린 시간을 저장할 변수
    kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);  // 칼만 필터 업데이트
    state_point = kf.get_x();  // 현재 상태 추정치 업데이트
    pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;  // LiDAR 위치 업데이트
    geoQuat.x = state_point.rot.coeffs()[0];  // 회전 쿼터니언 x 값 설정
    geoQuat.y = state_point.rot.coeffs()[1];  // 회전 쿼터니언 y 값 설정
    geoQuat.z = state_point.rot.coeffs()[2];  // 회전 쿼터니언 z 값 설정
    geoQuat.w = state_point.rot.coeffs()[3];  // 회전 쿼터니언 w 값 설정

    /******* 위치 추정 정보 퍼블리시 (odometry) *******/
    if (publish_tf_results) publish_visionpose(pubMavrosVisionPose);  // vision pose 퍼블리시
    publish_odometry(pubOdomAftMapped);  // odometry 퍼블리시

    /*** 특징 포인트를 kdtree에 추가 ***/
    map_incremental();  // 맵에 점을 추가하고 업데이트

    if(0)  // 맵 포인트를 확인해야 할 경우, 0을 1로 변경
    {
        PointVector().swap(ikdtree.PCL_Storage);  // 임시 저장소로 맵 포인트 교체
        ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);  // 맵을 플랫하게 변환
        featsFromMap->clear();  // 맵에서 가져온 포인트 클리어
        featsFromMap->points = ikdtree.PCL_Storage;  // 변환한 포인트를 맵에 추가
    }

    /******* 포인트 퍼블리시 *******/
    if (path_en)                         publish_path(pubPath);  // 경로 퍼블리시
    if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFull, pubLaserCloudFullTransformed);  // 월드 프레임 포인트 퍼블리시
    if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);  // 바디 프레임 포인트 퍼블리시
    // publish_map(pubLaserCloudMap);  // 맵 퍼블리시 (필요 시)

    /*** 번들 업데이트 또는 비동기 업데이트 ***/
    if (multi_lidar)  // 멀티 LiDAR 사용 시
    {
        // 다운샘플링된 포인트가 적고 효과적인 포인트 비율이 임계값보다 낮으면 번들 처리 활성화
        if (feats_down_size < voxelized_pt_num_thres && (double)effect_feat_num / feats_down_size < effect_pt_num_ratio_thres)
        {
            bundle_enabled = true;  // 번들 처리 활성화
            bundle_enabled_tic = 0;  // 번들 처리 틱 초기화
        }
        else if (bundle_enabled_tic > bundle_enabled_tic_thres)
        {
            bundle_enabled = false;  // 번들 처리 비활성화
        }
        bundle_enabled_tic++;  // 번들 처리 틱 증가
    }

    // 처리 시간 측정
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(t2 - t1).count() / 1000.0;  // 소요 시간(ms) 계산
    std_msgs::Float32 calc_time;
    calc_time.data = duration;
    pubCaclTime.publish(calc_time);  // 계산 시간 퍼블리시

    // 포인트 수 퍼블리시
    std_msgs::Float32 point_num;
    point_num.data = feats_down_size;
    pubPointNum.publish(point_num);

    // localizability 퍼블리시 (x, y, z 축 각각 퍼블리시)
    std_msgs::Float32 localizability_x, localizability_y, localizability_z;
    localizability_x.data = localizability_vec(0);
    localizability_y.data = localizability_vec(1);
    localizability_z.data = localizability_vec(2);
    pubLocalizabilityX.publish(localizability_x);
    pubLocalizabilityY.publish(localizability_y);
    pubLocalizabilityZ.publish(localizability_z);
}

// ROS 실행 상태를 확인하여 루프를 지속적으로 실행
status = ros::ok();
rate.sleep();  // 설정된 속도(5000Hz)로 루프 실행
}

    /**************** 맵 저장 ****************/
    /* 1. 충분한 메모리가 있는지 확인하세요
    /* 2. pcd 파일 저장은 실시간 성능에 큰 영향을 미칠 수 있습니다 **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)  // 저장할 포인트가 있고, 저장 기능이 활성화되어 있을 경우
    {
        string file_name = string("scans.pcd");  // 파일 이름 설정
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);  // 저장 경로 설정
        pcl::PCDWriter pcd_writer;  // PCD 파일 작성기 초기화
        cout << "현재 스캔이 /PCD/ 경로에 저장되었습니다: " << file_name << endl;  // 저장 알림 출력
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);  // 포인트 클라우드를 바이너리 형식으로 저장
    }

    return 0;  // 프로그램 종료
}

