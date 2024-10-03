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


/// c++ headers
#include <mutex>  // 뮤텍스 관련 헤더
#include <cmath>  // 수학 함수 관련 헤더
#include <csignal> // 신호 처리 관련 헤더
#include <unistd.h> // POSIX 운영체제 API 관련 헤더
#include <condition_variable> // 조건 변수 관련 헤더

/// module headers
#include <omp.h> // OpenMP 병렬 처리 관련 헤더

/// Eigen
#include <Eigen/Core> // Eigen 라이브러리의 기본 헤더

/// ros
#include <ros/ros.h> // ROS 기본 헤더
#include <geometry_msgs/Vector3.h> // 3D 벡터 메시지 형식
#include <geometry_msgs/PoseStamped.h> // 타임스탬프가 있는 포즈 메시지 형식
#include <nav_msgs/Odometry.h> // 위치 추정 메시지 형식
#include <nav_msgs/Path.h> // 경로 메시지 형식
#include <sensor_msgs/PointCloud2.h> // 포인트 클라우드 메시지 형식
#include <tf/transform_datatypes.h> // 변환 데이터 타입 관련 헤더
#include <tf/transform_broadcaster.h> // 변환 브로드캐스터 관련 헤더
#include <livox_ros_driver/CustomMsg.h> // Livox LiDAR의 커스텀 메시지 형식

/// pcl
#include <pcl/point_cloud.h> // PCL 포인트 클라우드 관련 헤더
#include <pcl/point_types.h> // PCL 포인트 타입 관련 헤더
#include <pcl/common/transforms.h> // 포인트 클라우드 변환 관련 함수 (transformPointCloud)
#include <pcl/filters/voxel_grid.h> // Voxel 그리드 필터 관련 헤더
#include <pcl_conversions/pcl_conversions.h> // PCL 데이터 변환 관련 헤더
#include <pcl/io/pcd_io.h> // PCD 파일 입출력 관련 헤더

/// this package
#include "so3_math.h" // SO3 수학 관련 헤더
#include "IMU_Processing.hpp" // IMU 데이터 처리 관련 헤더
#include "preprocess.h" // 전처리 관련 헤더
#include <ikd-Tree/ikd_Tree.h> // IKD 트리 관련 헤더
#include <chrono> // 시간 측정 관련 헤더
#include <std_msgs/Float32.h> // 32비트 부동 소수점 메시지 형식

using namespace std::chrono; // chrono 네임스페이스 사용

#define LASER_POINT_COV (0.001) // 레이저 포인트 공분산 정의


/**************************/
bool pcd_save_en = false; // PCD 파일 저장 기능 활성화 여부
bool extrinsic_est_en = true; // 외부 매개변수 추정 활성화 여부
bool path_en = true; // 경로 기능 활성화 여부

float res_last[100000] = {0.0}; // 마지막 결과를 저장할 배열
float DET_RANGE = 300.0f; // 감지 범위 설정
const float MOV_THRESHOLD = 1.5f; // 이동 임계값 설정

string lid_topic, lid_topic2, imu_topic; // LIDAR 및 IMU 토픽 이름
string map_frame = "map"; // 맵 프레임 이름 설정
bool multi_lidar = false; // 다중 LIDAR 사용 여부
bool async_debug = false; // 비동기 디버그 활성화 여부
bool publish_tf_results = false; // 변환 결과 퍼블리시 여부
bool extrinsic_imu_to_lidars = false; // IMU에서 LIDAR로의 외부 매개변수 설정 여부

double res_mean_last = 0.05; // 마지막 결과 평균
double total_residual = 0.0; // 총 잔여값

double last_timestamp_lidar = 0; // 마지막 LIDAR 타임스탬프
double last_timestamp_lidar2 = 0; // 두 번째 LIDAR의 마지막 타임스탬프
double last_timestamp_imu = -1.0; // 마지막 IMU 타임스탬프
double gyr_cov = 0.1; // 자이로 공분산
double acc_cov = 0.1; // 가속도 공분산
double b_gyr_cov = 0.0001; // 바이어스 자이로 공분산
double b_acc_cov = 0.0001; // 바이어스 가속도 공분산
double filter_size_surf = 0; // 표면 필터 크기 설정

double cube_len = 0; // 큐브 길이
double lidar_end_time = 0; // LIDAR 종료 시간
double lidar_end_time2 = 0; // 두 번째 LIDAR 종료 시간
double first_lidar_time = 0.0; // 첫 번째 LIDAR 시간

int effect_feat_num = 0; // 효과적인 특징 수
int feats_down_size = 0; // 다운 샘플링된 특징 크기
int NUM_MAX_ITERATIONS = 0; // 최대 반복 횟수
int pcd_save_interval = -1; // PCD 저장 간격
int pcd_index = 0; // PCD 인덱스

bool point_selected_surf[100000] = {0}; // 선택된 표면 포인트 배열
bool lidar_pushed = false; // LIDAR 푸시 여부
bool flg_exit = false; // 종료 플래그

bool scan_pub_en = false; // 스캔 퍼블리시 활성화 여부
bool dense_pub_en = false; // 밀집 퍼블리시 활성화 여부
bool scan_body_pub_en = false; // 스캔 바디 퍼블리시 활성화 여부

vector<BoxPointType> cub_needrm; // 큐브 포인트 타입 벡터
vector<PointVector> Nearest_Points; // 최근접 포인트 벡터

deque<double> time_buffer; // 시간 버퍼
deque<double> time_buffer2; // 두 번째 시간 버퍼
deque<PointCloudXYZI::Ptr> lidar_buffer; // LIDAR 버퍼
deque<PointCloudXYZI::Ptr> lidar_buffer2; // 두 번째 LIDAR 버퍼
deque<sensor_msgs::Imu::ConstPtr> imu_buffer; // IMU 버퍼

mutex mtx_buffer; // 뮤텍스 버퍼
condition_variable sig_buffer; // 신호 조건 변수

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI()); // 맵에서 얻은 특징 포인트 클라우드
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI()); // 왜곡되지 않은 특징 포인트 클라우드
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI()); // 바디에서 다운 샘플링된 특징 포인트 클라우드
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI()); // 월드에서 다운 샘플링된 특징 포인트 클라우드
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1)); // 노멀 벡터 포인트 클라우드
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1)); // 레이저 클라우드 원본 포인트 클라우드
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1)); // 보정된 노멀 벡터 포인트 클라우드
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI()); // 저장 대기 포인트 클라우드

pcl::VoxelGrid<PointType> downSizeFilterSurf; // 다운 샘플링 필터
KD_TREE<PointType> ikdtree; // KD 트리
shared_ptr<Preprocess> p_pre(new Preprocess()); // 전처리 객체
shared_ptr<ImuProcess> p_imu(new ImuProcess()); // IMU 처리 객체

Eigen::Matrix4d LiDAR2_wrt_LiDAR1 = Eigen::Matrix4d::Identity(); // LiDAR2를 LiDAR1에 대한 변환 행렬
Eigen::Matrix4d LiDAR1_wrt_drone = Eigen::Matrix4d::Identity(); // LiDAR1을 드론에 대한 변환 행렬

/*** EKF inputs and output ***/
MeasureGroup Measures; // EKF에 필요한 측정값 그룹
esekfom::esekf<state_ikfom, 12, input_ikfom> kf; // 확장 칼만 필터 객체 생성
state_ikfom state_point; // EKF 상태 포인트
vect3 pos_lid; // LIDAR 위치 벡터

nav_msgs::Path path; // 경로 메시지
nav_msgs::Odometry odomAftMapped; // 맵 후의 위치 추정 메시지
geometry_msgs::Quaternion geoQuat; // 쿼터니언

BoxPointType LocalMap_Points; // 로컬 맵 포인트
bool Localmap_Initialized = false; // 로컬 맵 초기화 여부
bool first_lidar_scan_check = false; // 첫 번째 LIDAR 스캔 여부 확인
double lidar_mean_scantime = 0.0; // LIDAR 평균 스캔 시간
double lidar_mean_scantime2 = 0.0; // 두 번째 LIDAR 평균 스캔 시간
int scan_num = 0; // 스캔 번호
int scan_num2 = 0; // 두 번째 스캔 번호
Eigen::Vector3d localizability_vec = Eigen::Vector3d::Zero(); // 로컬라이저빌리티 벡터 초기화


void SigHandle(int sig) // 시그널 핸들러 함수
{
    flg_exit = true; // 종료 플래그 설정
    ROS_WARN("catch sig %d", sig); // 시그널 경고 메시지 출력
    sig_buffer.notify_all(); // 모든 스레드에 시그널 알림
}

void pointBodyToWorld(PointType const * const pi, PointType * const po) // 바디 좌표를 월드 좌표로 변환하는 함수
{
    V3D p_body(pi->x, pi->y, pi->z); // 바디 좌표를 V3D 벡터로 변환
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos); // 글로벌 좌표 계산

    po->x = p_global(0); // 변환된 x 좌표 저장
    po->y = p_global(1); // 변환된 y 좌표 저장
    po->z = p_global(2); // 변환된 z 좌표 저장
    po->intensity = pi->intensity; // 세기 값 복사
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po) // 바디 좌표를 월드 좌표로 변환하는 템플릿 함수
{
    V3D p_body(pi[0], pi[1], pi[2]); // 바디 좌표를 V3D 벡터로 변환
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos); // 글로벌 좌표 계산

    po[0] = p_global(0); // 변환된 x 좌표 저장
    po[1] = p_global(1); // 변환된 y 좌표 저장
    po[2] = p_global(2); // 변환된 z 좌표 저장
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po) // RGB 포인트의 바디 좌표를 월드 좌표로 변환하는 함수
{
    V3D p_body(pi->x, pi->y, pi->z); // 바디 좌표를 V3D 벡터로 변환
    V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos); // 글로벌 좌표 계산

    po->x = p_global(0); // 변환된 x 좌표 저장
    po->y = p_global(1); // 변환된 y 좌표 저장
    po->z = p_global(2); // 변환된 z 좌표 저장
    po->intensity = pi->intensity; // 세기 값 복사
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po) // RGB 포인트의 LIDAR 바디 좌표를 IMU 좌표로 변환하는 함수
{
    V3D p_body_lidar(pi->x, pi->y, pi->z); // LIDAR 바디 좌표를 V3D 벡터로 변환
    V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I); // IMU 좌표 계산

    po->x = p_body_imu(0); // 변환된 x 좌표 저장
    po->y = p_body_imu(1); // 변환된 y 좌표 저장
    po->z = p_body_imu(2); // 변환된 z 좌표 저장
    po->intensity = pi->intensity; // 세기 값 복사
}

void lasermap_fov_segment() // LIDAR 맵의 시야를 세분화하는 함수
{
    cub_needrm.clear(); // 큐브 포인트 벡터 초기화
    V3D pos_LiD = pos_lid; // LIDAR의 현재 위치 저장

    if (!Localmap_Initialized) { // 로컬 맵이 초기화되지 않은 경우
        for (int i = 0; i < 3; i++) { // x, y, z 축에 대해
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0; // 맵의 최소 정점 계산
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0; // 맵의 최대 정점 계산
        }
        Localmap_Initialized = true; // 로컬 맵 초기화 플래그 설정
        return; // 함수 종료
    }

    float dist_to_map_edge[3][2]; // 맵 경계까지의 거리 배열
    bool need_move = false; // 이동 필요 여부 플래그 초기화

    for (int i = 0; i < 3; i++) { // x, y, z 축에 대해
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]); // 최소 정점까지의 거리 계산
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]); // 최대 정점까지의 거리 계산

        // 이동이 필요한 경우 플래그 설정
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) 
            need_move = true;
    }

    if (!need_move) return; // 이동이 필요하지 않으면 함수 종료

    BoxPointType New_LocalMap_Points, tmp_boxpoints; // 새로운 로컬 맵 포인트와 임시 박스 포인트 선언
    New_LocalMap_Points = LocalMap_Points; // 현재 로컬 맵 포인트 복사
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD - 1))); // 이동 거리 계산

    for (int i = 0; i < 3; i++) { // x, y, z 축에 대해
        tmp_boxpoints = LocalMap_Points; // 임시 박스 포인트에 현재 로컬 맵 포인트 저장

        // 최소 정점과의 거리 확인
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE) {
            New_LocalMap_Points.vertex_max[i] -= mov_dist; // 최대 정점 이동
            New_LocalMap_Points.vertex_min[i] -= mov_dist; // 최소 정점 이동
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist; // 임시 최소 정점 계산
            cub_needrm.push_back(tmp_boxpoints); // 큐브 포인트 추가
        } 
        // 최대 정점과의 거리 확인
        else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) {
            New_LocalMap_Points.vertex_max[i] += mov_dist; // 최대 정점 이동
            New_LocalMap_Points.vertex_min[i] += mov_dist; // 최소 정점 이동
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist; // 임시 최대 정점 계산
            cub_needrm.push_back(tmp_boxpoints); // 큐브 포인트 추가
        }
    }

    LocalMap_Points = New_LocalMap_Points; // 새로운 로컬 맵 포인트 업데이트

    if (cub_needrm.size() > 0) 
        ikdtree.Delete_Point_Boxes(cub_needrm); // 큐브 포인트가 있을 경우 KD 트리에서 삭제
}


void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) // 일반 포인트 클라우드 콜백 함수
{
    mtx_buffer.lock(); // 뮤텍스 잠금

    if (msg->header.stamp.toSec() < last_timestamp_lidar) // LIDAR 타임스탬프가 이전 타임스탬프보다 작은 경우
    {
        ROS_ERROR("lidar loop back, clear buffer"); // LIDAR 루프백 경고 메시지
        lidar_buffer.clear(); // LIDAR 버퍼 초기화
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI()); // 새로운 포인트 클라우드 포인터 생성
    p_pre->process(msg, ptr, 0); // 포인트 클라우드 전처리
    lidar_buffer.push_back(ptr); // 처리된 포인트 클라우드를 버퍼에 추가
    time_buffer.push_back(msg->header.stamp.toSec()); // 타임스탬프를 시간 버퍼에 추가
    last_timestamp_lidar = msg->header.stamp.toSec(); // 마지막 LIDAR 타임스탬프 업데이트
    mtx_buffer.unlock(); // 뮤텍스 잠금 해제
    sig_buffer.notify_all(); // 모든 스레드에 신호 알림
    first_lidar_scan_check = true; // 첫 번째 LIDAR 스캔 확인 플래그 설정
}

void standard_pcl_cbk2(const sensor_msgs::PointCloud2::ConstPtr &msg) // 두 번째 일반 포인트 클라우드 콜백 함수
{
    mtx_buffer.lock(); // 뮤텍스 잠금

    if (msg->header.stamp.toSec() < last_timestamp_lidar2) // LIDAR2 타임스탬프가 이전 타임스탬프보다 작은 경우
    {
        ROS_ERROR("lidar loop back, clear buffer"); // LIDAR 루프백 경고 메시지
        lidar_buffer2.clear(); // LIDAR2 버퍼 초기화
    }

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI()); // 새로운 포인트 클라우드 포인터 생성
    p_pre->process(msg, ptr, 1); // 포인트 클라우드 전처리
    lidar_buffer2.push_back(ptr); // 처리된 포인트 클라우드를 버퍼에 추가
    time_buffer2.push_back(msg->header.stamp.toSec()); // 타임스탬프를 시간 버퍼에 추가
    last_timestamp_lidar2 = msg->header.stamp.toSec(); // 마지막 LIDAR2 타임스탬프 업데이트
    mtx_buffer.unlock(); // 뮤텍스 잠금 해제
    sig_buffer.notify_all(); // 모든 스레드에 신호 알림
    first_lidar_scan_check = true; // 첫 번째 LIDAR 스캔 확인 플래그 설정
}

void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) // Livox 포인트 클라우드 콜백 함수
{
    mtx_buffer.lock(); // 뮤텍스 잠금

    if (msg->header.stamp.toSec() < last_timestamp_lidar) // LIDAR 타임스탬프가 이전 타임스탬프보다 작은 경우
    {
        ROS_ERROR("lidar loop back, clear buffer"); // LIDAR 루프백 경고 메시지
        lidar_buffer.clear(); // LIDAR 버퍼 초기화
    }
    last_timestamp_lidar = msg->header.stamp.toSec(); // 마지막 LIDAR 타임스탬프 업데이트

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI()); // 새로운 포인트 클라우드 포인터 생성
    p_pre->process(msg, ptr, 0); // 포인트 클라우드 전처리
    lidar_buffer.push_back(ptr); // 처리된 포인트 클라우드를 버퍼에 추가
    time_buffer.push_back(last_timestamp_lidar); // 타임스탬프를 시간 버퍼에 추가

    mtx_buffer.unlock(); // 뮤텍스 잠금 해제
    sig_buffer.notify_all(); // 모든 스레드에 신호 알림
    first_lidar_scan_check = true; // 첫 번째 LIDAR 스캔 확인 플래그 설정
}

void livox_pcl_cbk2(const livox_ros_driver::CustomMsg::ConstPtr &msg) // 두 번째 Livox 포인트 클라우드 콜백 함수
{
    mtx_buffer.lock(); // 뮤텍스 잠금

    if (msg->header.stamp.toSec() < last_timestamp_lidar2) // LIDAR2 타임스탬프가 이전 타임스탬프보다 작은 경우
    {
        ROS_ERROR("lidar loop back, clear buffer"); // LIDAR 루프백 경고 메시지
        lidar_buffer2.clear(); // LIDAR2 버퍼 초기화
    }
    last_timestamp_lidar2 = msg->header.stamp.toSec(); // 마지막 LIDAR2 타임스탬프 업데이트
    
    PointCloudXYZI::Ptr ptr(new PointCloudXYZI()); // 새로운 포인트 클라우드 포인터 생성
    p_pre->process(msg, ptr, 1); // 포인트 클라우드 전처리
    lidar_buffer2.push_back(ptr); // 처리된 포인트 클라우드를 버퍼에 추가
    time_buffer2.push_back(last_timestamp_lidar2); // 타임스탬프를 시간 버퍼에 추가
    
    mtx_buffer.unlock(); // 뮤텍스 잠금 해제
    sig_buffer.notify_all(); // 모든 스레드에 신호 알림
    first_lidar_scan_check = true; // 첫 번째 LIDAR 스캔 확인 플래그 설정
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) // IMU 콜백 함수
{
    if (!first_lidar_scan_check) return; // 첫 번째 LIDAR 스캔이 확인되지 않았으면 함수 종료 (IMU 입력만 스택되고 필터가 너무 많이 전파되는 것을 방지)

    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in)); // 입력 메시지를 복사하여 새로운 IMU 메시지 생성
    double timestamp = msg->header.stamp.toSec(); // 타임스탬프를 초 단위로 변환하여 저장

    mtx_buffer.lock(); // 뮤텍스 잠금
    if (timestamp < last_timestamp_imu) // IMU 타임스탬프가 이전 타임스탬프보다 작은 경우
    {
        ROS_WARN("imu loop back, clear buffer"); // IMU 루프백 경고 메시지 출력
        imu_buffer.clear(); // IMU 버퍼 초기화
    }

    last_timestamp_imu = timestamp; // 마지막 IMU 타임스탬프 업데이트

    imu_buffer.push_back(msg); // IMU 메시지를 버퍼에 추가
    mtx_buffer.unlock(); // 뮤텍스 잠금 해제
    sig_buffer.notify_all(); // 모든 스레드에 신호 알림
}

bool sync_packages(MeasureGroup &meas) // 측정 패키지를 동기화하는 함수
{
    if (multi_lidar) // 다중 LIDAR가 활성화된 경우
    {
        if (lidar_buffer.empty() || lidar_buffer2.empty() || imu_buffer.empty()) {
            return false; // 버퍼가 비어있으면 false 반환
        }

        /*** push a lidar scan ***/
        if (!lidar_pushed) // LIDAR 데이터가 푸시되지 않은 경우
        {
            meas.lidar = lidar_buffer.front(); // 첫 번째 LIDAR 포인트 클라우드를 측정 그룹에 추가
            meas.lidar_beg_time = time_buffer.front(); // LIDAR 시작 시간 저장
            
            // 포인트 클라우드가 너무 적은 경우 처리
            if (meas.lidar->points.size() <= 1) 
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime; // LIDAR 종료 시간 설정
                ROS_WARN("Too few input point cloud!\n"); // 경고 메시지 출력
            }
            else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime; // 평균 스캔 시간 추가
            }
            else // 정상적인 경우
            {
                scan_num++; // 스캔 번호 증가
                lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000); // 종료 시간 설정
                lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num; // 평균 스캔 시간 업데이트
            }
            meas.lidar_end_time = lidar_end_time; // 종료 시간 저장

            meas.lidar2 = lidar_buffer2.front(); // 두 번째 LIDAR 포인트 클라우드를 측정 그룹에 추가
            pcl::transformPointCloud(*meas.lidar2, *meas.lidar2, LiDAR2_wrt_LiDAR1); // LIDAR2 데이터를 LIDAR1 프레임으로 변환
            meas.lidar_beg_time2 = time_buffer2.front(); // 두 번째 LIDAR 시작 시간 저장
            
            // 두 번째 포인트 클라우드가 너무 적은 경우 처리
            if (meas.lidar2->points.size() <= 1) 
            {
                lidar_end_time2 = meas.lidar_beg_time2 + lidar_mean_scantime2; // 종료 시간 설정
                ROS_WARN("Too few input point cloud!\n"); // 경고 메시지 출력
            }
            else if (meas.lidar2->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime2)
            {
                lidar_end_time2 = meas.lidar_beg_time2 + lidar_mean_scantime2; // 평균 스캔 시간 추가
            }
            else // 정상적인 경우
            {
                scan_num2++; // 두 번째 스캔 번호 증가
                lidar_end_time2 = meas.lidar_beg_time2 + meas.lidar2->points.back().curvature / double(1000); // 종료 시간 설정
                lidar_mean_scantime2 += (meas.lidar2->points.back().curvature / double(1000) - lidar_mean_scantime2) / scan_num2; // 평균 스캔 시간 업데이트
            }
            meas.lidar_end_time2 = lidar_end_time2; // 종료 시간 저장

            lidar_pushed = true; // LIDAR 푸시 플래그 설정
        }

        if (last_timestamp_imu < lidar_end_time || last_timestamp_imu < lidar_end_time2)
        {
            return false; // IMU 타임스탬프가 LIDAR 종료 시간보다 작으면 false 반환
        }

        /*** push imu data, and pop from imu buffer ***/
        double imu_time = imu_buffer.front()->header.stamp.toSec(); // IMU 타임스탬프
        meas.imu.clear(); // IMU 측정값 초기화
        while ((!imu_buffer.empty()) && (imu_time < lidar_end_time || imu_time < lidar_end_time2))
        {
            imu_time = imu_buffer.front()->header.stamp.toSec(); // IMU 타임스탬프 업데이트
            if (imu_time > lidar_end_time && imu_time > lidar_end_time2) break; // 종료 시간보다 크면 반복 종료
            meas.imu.push_back(imu_buffer.front()); // IMU 데이터를 측정 그룹에 추가
            imu_buffer.pop_front(); // IMU 버퍼에서 제거
        }

        lidar_buffer.pop_front(); // LIDAR 버퍼에서 제거
        time_buffer.pop_front(); // 시간 버퍼에서 제거
        lidar_buffer2.pop_front(); // 두 번째 LIDAR 버퍼에서 제거
        time_buffer2.pop_front(); // 두 번째 시간 버퍼에서 제거

        lidar_pushed = false; // LIDAR 푸시 플래그 초기화
        return true; // 성공적으로 동기화된 경우 true 반환
    }
    else // 다중 LIDAR가 비활성화된 경우
    {
        if (lidar_buffer.empty() || imu_buffer.empty()) {
            return false; // 버퍼가 비어있으면 false 반환
        }
        
        /*** push a lidar scan ***/
        if (!lidar_pushed) // LIDAR 데이터가 푸시되지 않은 경우
        {
            meas.lidar = lidar_buffer.front(); // 첫 번째 LIDAR 포인트 클라우드를 측정 그룹에 추가
            meas.lidar_beg_time = time_buffer.front(); // LIDAR 시작 시간 저장
            
            // 포인트 클라우드가 너무 적은 경우 처리
            if (meas.lidar->points.size() <= 1) 
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime; // LIDAR 종료 시간 설정
                ROS_WARN("Too few input point cloud!\n"); // 경고 메시지 출력
            }
            else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime; // 평균 스캔 시간 추가
            }
            else // 정상적인 경우
            {
                scan_num++; // 스캔 번호 증가
                lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000); // 종료 시간 설정
                lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num; // 평균 스캔 시간 업데이트
            }

            meas.lidar_end_time = lidar_end_time; // 종료 시간 저장

            lidar_pushed = true; // LIDAR 푸시 플래그 설정
        }

        if (last_timestamp_imu < lidar_end_time) // IMU 타임스탬프가 LIDAR 종료 시간보다 작은 경우
        {
            return false; // false 반환
        }

        /*** push imu data, and pop from imu buffer ***/
        double imu_time = imu_buffer.front()->header.stamp.toSec(); // IMU 타임스탬프
        meas.imu.clear(); // IMU 측정값 초기화
        while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
        {
            imu_time = imu_buffer.front()->header.stamp.toSec(); // IMU 타임스탬프 업데이트
            if (imu_time > lidar_end_time) break; // 종료 시간보다 크면 반복 종료
            meas.imu.push_back(imu_buffer.front()); // IMU 데이터를 측정 그룹에 추가
            imu_buffer.pop_front(); // IMU 버퍼에서 제거
        }

        lidar_buffer.pop_front(); // LIDAR 버퍼에서 제거
        time_buffer.pop_front(); // 시간 버퍼에서 제거

        lidar_pushed = false; // LIDAR 푸시 플래그 초기화
        return true; // 성공적으로 동기화된 경우 true 반환        
    }
}
void map_incremental() // 점진적으로 맵을 업데이트하는 함수
{
    PointVector PointToAdd; // 추가할 포인트 벡터
    PointVector PointNoNeedDownsample; // 다운샘플링이 필요 없는 포인트 벡터
    PointToAdd.reserve(feats_down_size); // 추가할 포인트 벡터의 용량 예약
    PointNoNeedDownsample.reserve(feats_down_size); // 필요 없는 포인트 벡터의 용량 예약

    for (int i = 0; i < feats_down_size; i++) // 다운샘플링된 포인트 수 만큼 반복
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i])); // 바디 좌표를 월드 좌표로 변환

        /* decide if need add to map */
        if (!Nearest_Points[i].empty()) // 최근접 포인트가 있는 경우
        {
            const PointVector &points_near = Nearest_Points[i]; // 최근접 포인트 벡터 저장
            bool need_add = true; // 추가 필요 여부 플래그 초기화
            BoxPointType Box_of_Point; // 박스 포인트 타입 선언
            PointType downsample_result, mid_point; // 다운샘플링 결과 및 중간 포인트 선언
            
            // 중간 포인트 계산
            mid_point.x = floor(feats_down_world->points[i].x / filter_size_surf) * filter_size_surf + 0.5 * filter_size_surf;
            mid_point.y = floor(feats_down_world->points[i].y / filter_size_surf) * filter_size_surf + 0.5 * filter_size_surf;
            mid_point.z = floor(feats_down_world->points[i].z / filter_size_surf) * filter_size_surf + 0.5 * filter_size_surf;
            
            float dist = calc_dist(feats_down_world->points[i], mid_point); // 거리 계산
            
            // 최근접 포인트와의 거리 비교
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_surf && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_surf && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_surf) {
                PointNoNeedDownsample.push_back(feats_down_world->points[i]); // 다운샘플링 필요 없는 포인트에 추가
                continue; // 다음 반복으로 이동
            }
            
            // NUM_MATCH_POINTS 만큼 반복
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++) 
            {
                if (points_near.size() < NUM_MATCH_POINTS) break; // 최근접 포인트 수가 NUM_MATCH_POINTS보다 작으면 반복 종료
                if (calc_dist(points_near[readd_i], mid_point) < dist) // 중간 포인트와의 거리 비교
                {
                    need_add = false; // 추가 필요 없음을 표시
                    break; // 반복 종료
                }
            }

            if (need_add) PointToAdd.push_back(feats_down_world->points[i]); // 추가가 필요하면 포인트를 추가
        }
        else // 최근접 포인트가 없는 경우
        {
            PointToAdd.push_back(feats_down_world->points[i]); // 포인트를 추가
        }
    }
    
    ikdtree.Add_Points(PointToAdd, true); // 추가할 포인트를 KD 트리에 추가
    ikdtree.Add_Points(PointNoNeedDownsample, false); // 필요 없는 포인트를 KD 트리에 추가
    return; // 함수 종료
}

void publish_frame_world(const ros::Publisher &pubLaserCloudFull, const ros::Publisher &pubLaserCloudFullTransFormed) // 월드 프레임을 퍼블리시하는 함수
{
    if (scan_pub_en) // 스캔 퍼블리시가 활성화된 경우
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body); // 선택된 포인트 클라우드 포인터
        int size = laserCloudFullRes->points.size(); // 포인트 클라우드의 크기
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1)); // 월드 좌표 포인트 클라우드 생성

        for (int i = 0; i < size; i++) // 포인트 클라우드의 각 포인트에 대해
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]); // 바디 좌표를 월드 좌표로 변환
        }

        sensor_msgs::PointCloud2 laserCloudmsg; // ROS 메시지 타입 선언
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg); // 포인트 클라우드를 ROS 메시지로 변환
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time); // 타임스탬프 설정
        laserCloudmsg.header.frame_id = map_frame; // 프레임 ID 설정
        pubLaserCloudFull.publish(laserCloudmsg); // 퍼블리시

        if (publish_tf_results) // TF 결과 퍼블리시가 활성화된 경우
        {
            PointCloudXYZI::Ptr laserCloudWorldTransFormed(new PointCloudXYZI(size, 1)); // 변환된 포인트 클라우드 생성
            pcl::transformPointCloud(*laserCloudWorld, *laserCloudWorldTransFormed, LiDAR1_wrt_drone); // 포인트 클라우드 변환
            sensor_msgs::PointCloud2 laserCloudmsg2; // ROS 메시지 타입 선언
            pcl::toROSMsg(*laserCloudWorldTransFormed, laserCloudmsg2); // 변환된 포인트 클라우드를 ROS 메시지로 변환
            laserCloudmsg2.header.stamp = ros::Time().fromSec(lidar_end_time); // 타임스탬프 설정
            laserCloudmsg2.header.frame_id = map_frame; // 프레임 ID 설정
            pubLaserCloudFullTransFormed.publish(laserCloudmsg2); // 퍼블리시
        }
    }
}

/**************** 맵 저장 ****************/
/* 1. 충분한 메모리가 있는지 확인하세요
/* 2. PCD 저장이 실시간 성능에 영향을 미칠 수 있음을 유의하세요 **/
if (pcd_save_en) // PCD 저장 기능이 활성화된 경우
{
    int size = feats_undistort->points.size(); // 왜곡되지 않은 포인트 클라우드의 포인트 수
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1)); // 월드 좌표 포인트 클라우드 생성

    for (int i = 0; i < size; i++) // 각 포인트에 대해
    {
        RGBpointBodyToWorld(&feats_undistort->points[i], // 바디 좌표를 월드 좌표로 변환
                            &laserCloudWorld->points[i]);
    }
    
    *pcl_wait_save += *laserCloudWorld; // 변환된 포인트 클라우드를 대기 포인트 클라우드에 추가

    static int scan_wait_num = 0; // 대기 스캔 번호 초기화
    scan_wait_num++; // 대기 스캔 번호 증가
    
    // 저장 조건 확인
    if (pcl_wait_save->size() > 0 && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
    {
        pcd_index++; // PCD 인덱스 증가
        string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd")); // 저장 경로 생성
        pcl::PCDWriter pcd_writer; // PCD 작성기 생성
        cout << "current scan saved to /PCD/" << all_points_dir << endl; // 현재 스캔 저장 메시지 출력
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save); // PCD 파일로 저장
        pcl_wait_save->clear(); // 대기 포인트 클라우드 초기화
        scan_wait_num = 0; // 대기 스캔 번호 초기화
    }
  }
}

void publish_frame_body(const ros::Publisher &pubLaserCloudFull_body) // 바디 프레임을 퍼블리시하는 함수
{
    int size = feats_undistort->points.size(); // 왜곡되지 않은 포인트 클라우드의 포인트 수
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1)); // IMU 바디 좌표 포인트 클라우드 생성

    for (int i = 0; i < size; i++) // 각 포인트에 대해
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], // LIDAR 바디 좌표를 IMU 좌표로 변환
                               &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg; // ROS 메시지 타입 선언
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg); // 포인트 클라우드를 ROS 메시지로 변환
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time); // 타임스탬프 설정
    laserCloudmsg.header.frame_id = "body"; // 프레임 ID 설정
    pubLaserCloudFull_body.publish(laserCloudmsg); // 퍼블리시
}

void publish_map(const ros::Publisher &pubLaserCloudMap) // 맵을 퍼블리시하는 함수
{
    sensor_msgs::PointCloud2 laserCloudMap; // ROS 메시지 타입 선언
    pcl::toROSMsg(*featsFromMap, laserCloudMap); // 포인트 클라우드를 ROS 메시지로 변환
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time); // 타임스탬프 설정
    laserCloudMap.header.frame_id = map_frame; // 프레임 ID 설정
    pubLaserCloudMap.publish(laserCloudMap); // 퍼블리시
}

template<typename T>
void set_posestamp(T &out) // 포즈 타임스탬프 설정하는 템플릿 함수
{
    out.pose.position.x = state_point.pos(0); // x 좌표 설정
    out.pose.position.y = state_point.pos(1); // y 좌표 설정
    out.pose.position.z = state_point.pos(2); // z 좌표 설정
    out.pose.orientation.x = geoQuat.x; // 쿼터니언 x 설정
    out.pose.orientation.y = geoQuat.y; // 쿼터니언 y 설정
    out.pose.orientation.z = geoQuat.z; // 쿼터니언 z 설정
    out.pose.orientation.w = geoQuat.w; // 쿼터니언 w 설정
    return; // 함수 종료
}

void publish_visionpose(const ros::Publisher &publisher) // 비전 포즈를 퍼블리시하는 함수
{
    geometry_msgs::PoseStamped msg_out_; // 출력 포즈 메시지 선언
    msg_out_.header.frame_id = map_frame; // 프레임 ID 설정
    msg_out_.header.stamp = ros::Time().fromSec(lidar_end_time); // 타임스탬프 설정

    Eigen::Matrix4d current_pose_eig_ = Eigen::Matrix4d::Identity(); // 현재 포즈의 행렬 초기화
    current_pose_eig_.block<3, 3>(0, 0) = state_point.rot.toRotationMatrix(); // 회전 행렬 설정
    current_pose_eig_.block<3, 1>(0, 3) = state_point.pos; // 위치 설정
    
    // 드론에 대한 LIDAR1의 변환 계산
    Eigen::Matrix4d tfed_vision_pose_eig_ = LiDAR1_wrt_drone * current_pose_eig_ * LiDAR1_wrt_drone.inverse(); // 참고

    // 변환된 포즈의 위치 설정
    msg_out_.pose.position.x = tfed_vision_pose_eig_(0, 3);
    msg_out_.pose.position.y = tfed_vision_pose_eig_(1, 3);
    msg_out_.pose.position.z = tfed_vision_pose_eig_(2, 3);
    
    // 변환된 포즈의 쿼터니언 설정
    Eigen::Quaterniond tfed_quat_(tfed_vision_pose_eig_.block<3, 3>(0, 0)); // 회전 행렬에서 쿼터니언 생성
    msg_out_.pose.orientation.x = tfed_quat_.x();
    msg_out_.pose.orientation.y = tfed_quat_.y();
    msg_out_.pose.orientation.z = tfed_quat_.z();
    msg_out_.pose.orientation.w = tfed_quat_.w();
    
    publisher.publish(msg_out_); // 퍼블리시
    return; // 함수 종료
}
void publish_odometry(const ros::Publisher &pubOdomAftMapped) // 오도메트리 정보를 퍼블리시하는 함수
{
    odomAftMapped.header.frame_id = map_frame; // 프레임 ID 설정
    odomAftMapped.child_frame_id = "body"; // 자식 프레임 ID 설정
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time); // 타임스탬프 설정
    set_posestamp(odomAftMapped.pose); // 포즈 타임스탬프 설정
    pubOdomAftMapped.publish(odomAftMapped); // 오도메트리 메시지 퍼블리시

    auto P = kf.get_P(); // 칼만 필터의 공분산 행렬 가져오기
    for (int i = 0; i < 6; i++) // 6개의 공분산 요소 설정
    {
        int k = i < 3 ? i + 3 : i - 3; // 인덱스 변환
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3); // 공분산 행렬 요소 설정
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br_odom_to_body; // 변환 브로드캐스터 선언
    tf::Transform transform; // 변환 객체 선언
    tf::Quaternion q; // 쿼터니언 객체 선언
    
    // 변환의 원점 설정
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, 
                                     odomAftMapped.pose.pose.position.y, 
                                     odomAftMapped.pose.pose.position.z));
    // 쿼터니언 설정
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation(q); // 변환에 쿼터니언 설정
    br_odom_to_body.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, map_frame, "body")); // 변환 송신
}

void publish_path(const ros::Publisher pubPath) // 경로 정보를 퍼블리시하는 함수
{
    geometry_msgs::PoseStamped msg_body_pose; // 포즈 스탬프 메시지 선언
    set_posestamp(msg_body_pose); // 포즈 타임스탬프 설정
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time); // 타임스탬프 설정
    msg_body_pose.header.frame_id = map_frame; // 프레임 ID 설정

    /*** 경로가 너무 커지면 RViz가 충돌할 수 있음 ***/
    static int jjj = 0; // 카운터 초기화
    jjj++; // 카운터 증가
    if (jjj % 10 == 0) // 10회마다 경로에 추가
    {
        path.poses.push_back(msg_body_pose); // 경로에 포즈 추가
        pubPath.publish(path); // 경로 퍼블리시
    }
}



void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) // 공유 모델 계산 함수
{
    laserCloudOri->clear(); // 원래 레이저 클라우드 초기화
    corr_normvect->clear(); // 보정된 노멀 벡터 초기화
    total_residual = 0.0; // 총 잔여값 초기화

    /** 가장 가까운 표면 검색 및 잔여값 계산 **/
    #ifdef MP_EN // 다중 처리 활성화
        omp_set_num_threads(MP_PROC_NUM); // 스레드 수 설정
        #pragma omp parallel for // 병렬 처리 시작
    #endif
    for (int i = 0; i < feats_down_size; i++) // 다운샘플링된 포인트 수 만큼 반복
    {
        PointType &point_body  = feats_down_body->points[i]; // 바디 포인트 가져오기
        PointType &point_world = feats_down_world->points[i]; // 월드 포인트 가져오기

        /* 월드 프레임으로 변환 */
        V3D p_body(point_body.x, point_body.y, point_body.z); // 바디 포인트를 V3D 벡터로 변환
        V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos); // 글로벌 좌표 계산
        point_world.x = p_global(0); // 변환된 x 좌표 저장
        point_world.y = p_global(1); // 변환된 y 좌표 저장
        point_world.z = p_global(2); // 변환된 z 좌표 저장
        point_world.intensity = point_body.intensity; // 세기 값 복사

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS); // 최근접 포인트 거리 저장 벡터
        auto &points_near = Nearest_Points[i]; // 최근접 포인트 저장

        if (ekfom_data.converge) // 수렴 상태일 경우
        {
            /** 맵에서 가장 가까운 표면 찾기 **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis); // 최근접 포인트 검색
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true; // 선택된 표면 여부 결정
        }

        if (!point_selected_surf[i]) continue; // 선택된 표면이 없는 경우 다음 반복으로 이동

        VF(4) pabcd; // 평면 계수 배열
        point_selected_surf[i] = false; // 초기 선택된 표면 플래그 설정
        if (esti_plane(pabcd, points_near, 0.1f)) // 평면 추정
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3); // 거리 계산
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm()); // 적합도 계산

            if (s > 0.9) // 적합도가 임계값을 넘는 경우
            {
                point_selected_surf[i] = true; // 선택된 표면 플래그 설정
                normvec->points[i].x = pabcd(0); // 노멀 벡터 설정
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2; // 잔여값 설정
                res_last[i] = abs(pd2); // 잔여값 저장
            }
        }
    }
    
    effect_feat_num = 0; // 효과적인 특징 수 초기화
    localizability_vec = Eigen::Vector3d::Zero(); // 로컬라이저빌리티 벡터 초기화
    for (int i = 0; i < feats_down_size; i++) // 다운샘플링된 포인트 수 만큼 반복
    {
        if (point_selected_surf[i]) // 선택된 표면이 있는 경우
        {
            laserCloudOri->points[effect_feat_num] = feats_down_body->points[i]; // 포인트 추가
            corr_normvect->points[effect_feat_num] = normvec->points[i]; // 보정된 노멀 벡터 추가
            total_residual += res_last[i]; // 총 잔여값 업데이트
            effect_feat_num++; // 효과적인 특징 수 증가
            localizability_vec += Eigen::Vector3d(normvec->points[i].x, normvec->points[i].y, normvec->points[i].z).array().square().matrix(); // 로컬라이저빌리티 벡터 업데이트
        }
    }
    localizability_vec = localizability_vec.cwiseSqrt(); // 로컬라이저빌리티 벡터 제곱근 계산

    if (effect_feat_num < 1) // 효과적인 특징 수가 1보다 작은 경우
    {
        ekfom_data.valid = false; // 유효성 플래그 설정
        ROS_WARN("No Effective Points! \n"); // 경고 메시지 출력
        return; // 함수 종료
    }

    res_mean_last = total_residual / effect_feat_num; // 평균 잔여값 계산
    
    /*** 측정 자코비안 행렬 H 및 측정 벡터 계산 ***/
    ekfom_data.h_x = MatrixXd::Zero(effect_feat_num, 12); // 자코비안 행렬 초기화
    ekfom_data.h.resize(effect_feat_num); // 측정 벡터 크기 설정

    for (int i = 0; i < effect_feat_num; i++) // 효과적인 특징 수 만큼 반복
    {
        const PointType &laser_p  = laserCloudOri->points[i]; // 레이저 포인트 가져오기
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z); // 포인트 V3D 벡터 변환
        M3D point_be_crossmat; // 바디 포인트의 교차 행렬
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be); // 교차 행렬 계산
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I; // 변환된 포인트 계산
        M3D point_crossmat; // 변환된 포인트의 교차 행렬
        point_crossmat << SKEW_SYM_MATRX(point_this); // 교차 행렬 계산

        /*** 가장 가까운 표면/모서리의 노멀 벡터 얻기 ***/
        const PointType &norm_p = corr_normvect->points[i]; // 보정된 노멀 벡터 가져오기
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z); // 노멀 벡터 V3D 변환

        /*** 측정 자코비안 행렬 H 계산 ***/
        V3D C(s.rot.conjugate() * norm_vec); // 회전 변환된 노멀 벡터
        V3D A(point_crossmat * C); // 교차 행렬을 통한 계산
        if (extrinsic_est_en) // 외부 매개변수 추정이 활성화된 경우
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); // 외부 매개변수 고려한 계산
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C); // 자코비안 행렬 설정
        }
        else // 외부 매개변수 추정이 비활성화된 경우
        {
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0; // 자코비안 행렬 설정
        }

        /*** 측정: 가장 가까운 표면/모서리까지의 거리 ***/
        ekfom_data.h(i) = -norm_p.intensity; // 측정값 설정
    }
}

int main(int argc, char** argv) // 메인 함수
{
    ros::init(argc, argv, "laserMapping"); // ROS 노드 초기화
    ros::NodeHandle nh; // 노드 핸들 생성

    // 임시 변수
    vector<double> extrinT(3, 0.0); // 외부 매개변수 T 초기화
    vector<double> extrinR(9, 0.0); // 외부 매개변수 R 초기화
    vector<double> extrinT2(3, 0.0); // 두 번째 외부 매개변수 T 초기화
    vector<double> extrinR2(9, 0.0); // 두 번째 외부 매개변수 R 초기화
    vector<double> extrinT3(3, 0.0); // 세 번째 외부 매개변수 T 초기화
    vector<double> extrinR3(9, 0.0); // 세 번째 외부 매개변수 R 초기화
    vector<double> extrinT4(3, 0.0); // 네 번째 외부 매개변수 T 초기화
    vector<double> extrinR4(9, 0.0); // 네 번째 외부 매개변수 R 초기화

    // 파라미터 설정
    nh.param<int>("common/max_iteration", NUM_MAX_ITERATIONS, 4); // 최대 반복 횟수
    nh.param<bool>("common/async_debug", async_debug, false); // 비동기 디버그 활성화 여부
    nh.param<bool>("common/multi_lidar", multi_lidar, true); // 다중 LIDAR 사용 여부
    nh.param<bool>("common/publish_tf_results", publish_tf_results, true); // TF 결과 퍼블리시 여부
    nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar"); // LIDAR 토픽 이름
    nh.param<string>("common/lid_topic2", lid_topic2, "/livox/lidar"); // 두 번째 LIDAR 토픽 이름
    nh.param<string>("common/imu_topic", imu_topic, "/livox/imu"); // IMU 토픽 이름
    nh.param<string>("common/map_frame", map_frame, "map"); // 맵 프레임 이름

    nh.param<double>("preprocess/filter_size_surf", filter_size_surf, 0.5); // 필터 크기 설정
    nh.param<int>("preprocess/point_filter_num", p_pre->point_filter_num[0], 2); // 포인트 필터 번호 설정
    nh.param<int>("preprocess/point_filter_num2", p_pre->point_filter_num[1], 2); // 두 번째 포인트 필터 번호 설정
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type[0], AVIA); // LIDAR 타입 설정
    nh.param<int>("preprocess/lidar_type2", p_pre->lidar_type[1], AVIA); // 두 번째 LIDAR 타입 설정
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS[0], 16); // 스캔 라인 수 설정
    nh.param<int>("preprocess/scan_line2", p_pre->N_SCANS[1], 16); // 두 번째 스캔 라인 수 설정
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE[0], 10); // 스캔 비율 설정
    nh.param<int>("preprocess/scan_rate2", p_pre->SCAN_RATE[1], 10); // 두 번째 스캔 비율 설정
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit[0], US); // 타임스탬프 단위 설정
    nh.param<int>("preprocess/timestamp_unit2", p_pre->time_unit[1], US); // 두 번째 타임스탬프 단위 설정
    nh.param<double>("preprocess/blind", p_pre->blind[0], 0.01); // 블라인드 설정
    nh.param<double>("preprocess/blind2", p_pre->blind[1], 0.01); // 두 번째 블라인드 설정
    p_pre->set(); // 전처리 설정

    nh.param<double>("mapping/cube_side_length", cube_len, 200.0); // 큐브 변의 길이 설정
    nh.param<float>("mapping/det_range", DET_RANGE, 300.f); // 감지 범위 설정
    nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1); // 자이로 공분산 설정
    nh.param<double>("mapping/acc_cov", acc_cov, 0.1); // 가속도 공분산 설정
    nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001); // 바이어스 자이로 공분산 설정
    nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001); // 바이어스 가속도 공분산 설정
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true); // 외부 매개변수 추정 활성화 여부
    nh.param<bool>("mapping/extrinsic_imu_to_lidars", extrinsic_imu_to_lidars, true); // IMU에서 LIDAR로의 외부 매개변수 설정 여부
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>()); // 외부 매개변수 T 설정
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>()); // 외부 매개변수 R 설정
    nh.param<vector<double>>("mapping/extrinsic_T2", extrinT2, vector<double>()); // 두 번째 외부 매개변수 T 설정
    nh.param<vector<double>>("mapping/extrinsic_R2", extrinR2, vector<double>()); // 두 번째 외부 매개변수 R 설정
    nh.param<vector<double>>("mapping/extrinsic_T_L2_wrt_L1", extrinT3, vector<double>()); // LIDAR2가 LIDAR1에 대한 외부 매개변수 T 설정
    nh.param<vector<double>>("mapping/extrinsic_R_L2_wrt_L1", extrinR3, vector<double>()); // LIDAR2가 LIDAR1에 대한 외부 매개변수 R 설정
    nh.param<vector<double>>("mapping/extrinsic_T_L1_wrt_drone", extrinT4, vector<double>()); // LIDAR1이 드론에 대한 외부 매개변수 T 설정
    nh.param<vector<double>>("mapping/extrinsic_R_L1_wrt_drone", extrinR4, vector<double>()); // LIDAR1이 드론에 대한 외부 매개변수 R 설정

    nh.param<bool>("publish/path_en", path_en, true); // 경로 퍼블리시 활성화 여부
    nh.param<bool>("publish/scan_publish_en", scan_pub_en, true); // 스캔 퍼블리시 활성화 여부
    nh.param<bool>("publish/dense_publish_en", dense_pub_en, true); // 밀집 퍼블리시 활성화 여부
    nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en, true); // 바디 프레임 퍼블리시 활성화 여부
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false); // PCD 저장 활성화 여부
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1); // PCD 저장 간격 설정



    /*** 변수 정의 ***/
    path.header.stamp = ros::Time::now(); // 현재 시간으로 타임스탬프 설정
    path.header.frame_id = map_frame; // 프레임 ID 설정
    memset(point_selected_surf, true, sizeof(point_selected_surf)); // 선택된 표면 배열을 true로 초기화
    memset(res_last, -1000.0f, sizeof(res_last)); // 마지막 잔여값 배열을 -1000.0f로 초기화
    downSizeFilterSurf.setLeafSize(filter_size_surf, filter_size_surf, filter_size_surf); // 다운샘플링 필터의 리프 사이즈 설정
    ikdtree.set_downsample_param(filter_size_surf); // KD 트리 다운샘플링 파라미터 설정
    
    V3D Lidar_T_wrt_IMU(Zero3d); // IMU에 대한 LIDAR의 위치 초기화
    M3D Lidar_R_wrt_IMU(Eye3d); // IMU에 대한 LIDAR의 회전 초기화
    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT); // 외부 매개변수 T 설정
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR); // 외부 매개변수 R 설정
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU); // IMU에 외부 매개변수 설정
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov)); // 자이로 공분산 설정
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov)); // 가속도 공분산 설정
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov)); // 자이로 바이어스 공분산 설정
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov)); // 가속도 바이어스 공분산 설정

  // 다중 LIDAR TF 전용
    if (multi_lidar) // 다중 LIDAR가 활성화된 경우
    {
        if (extrinsic_imu_to_lidars) // IMU에서 LIDAR로의 외부 매개변수 설정 여부 확인
        {
            Eigen::Matrix4d Lidar_wrt_IMU = Eigen::Matrix4d::Identity(); // IMU에 대한 LIDAR의 변환 행렬 초기화
            Eigen::Matrix4d Lidar2_wrt_IMU = Eigen::Matrix4d::Identity(); // 두 번째 LIDAR의 IMU에 대한 변환 행렬 초기화
            V3D LiDAR2_T_wrt_IMU; 
            LiDAR2_T_wrt_IMU << VEC_FROM_ARRAY(extrinT2); // 두 번째 LIDAR의 위치 설정
            M3D LiDAR2_R_wrt_IMU; 
            LiDAR2_R_wrt_IMU << MAT_FROM_ARRAY(extrinR2); // 두 번째 LIDAR의 회전 설정
            Lidar_wrt_IMU.block<3,3>(0,0) = Lidar_R_wrt_IMU; // LIDAR의 회전 행렬 설정
            Lidar_wrt_IMU.block<3,1>(0,3) = Lidar_T_wrt_IMU; // LIDAR의 위치 설정
            Lidar2_wrt_IMU.block<3,3>(0,0) = LiDAR2_R_wrt_IMU; // 두 번째 LIDAR의 회전 행렬 설정
            Lidar2_wrt_IMU.block<3,1>(0,3) = LiDAR2_T_wrt_IMU; // 두 번째 LIDAR의 위치 설정
            LiDAR2_wrt_LiDAR1 = Lidar_wrt_IMU.inverse() * Lidar2_wrt_IMU; // LIDAR2를 LIDAR1에 대한 변환 계산
        }
        else // IMU에서 LIDAR로의 외부 매개변수 설정이 비활성화된 경우
        {
            V3D LiDAR2_T_wrt_LiDAR1; 
            LiDAR2_T_wrt_LiDAR1 << VEC_FROM_ARRAY(extrinT3); // LIDAR2가 LIDAR1에 대한 위치 설정
            M3D Lidar2_R_wrt_LiDAR1; 
            Lidar2_R_wrt_LiDAR1 << MAT_FROM_ARRAY(extrinR3); // LIDAR2가 LIDAR1에 대한 회전 설정
            LiDAR2_wrt_LiDAR1.block<3,3>(0,0) = Lidar2_R_wrt_LiDAR1; // LIDAR2의 회전 행렬 설정
            LiDAR2_wrt_LiDAR1.block<3,1>(0,3) = LiDAR2_T_wrt_LiDAR1; // LIDAR2의 위치 설정
        }
        cout << "\033[32;1mMulti LiDAR on!" << endl; // 다중 LIDAR 활성화 메시지 출력
        cout << "lidar_type[0]: " << p_pre->lidar_type[0] << ", " << "lidar_type[1]: " << p_pre->lidar_type[1] << endl << endl; // LIDAR 타입 출력
        cout << "L2 wrt L1 TF: " << endl << LiDAR2_wrt_LiDAR1 << "\033[0m" << endl << endl; // LIDAR2에 대한 LIDAR1의 변환 행렬 출력
    }
    
    if (publish_tf_results) // TF 결과 퍼블리시 여부 확인
    {
        V3D LiDAR1_T_wrt_drone; 
        LiDAR1_T_wrt_drone << VEC_FROM_ARRAY(extrinT4); // LIDAR1이 드론에 대한 위치 설정
        M3D LiDAR2_R_wrt_drone; 
        LiDAR2_R_wrt_drone << MAT_FROM_ARRAY(extrinR4); // LIDAR1이 드론에 대한 회전 설정
        LiDAR1_wrt_drone.block<3,3>(0,0) = LiDAR2_R_wrt_drone; // LIDAR1의 회전 행렬 설정
        LiDAR1_wrt_drone.block<3,1>(0,3) = LiDAR1_T_wrt_drone; // LIDAR1의 위치 설정
        cout << "\033[32;1mLiDAR wrt Drone:" << endl; // LIDAR 드론 변환 메시지 출력
        cout << LiDAR1_wrt_drone << "\033[0m" << endl << endl; // LIDAR 드론 변환 행렬 출력
    }
    
    // EPSI 초기화
    double epsi[23] = {0.001}; // 초기 잔차 값 설정
    fill(epsi, epsi + 23, 0.001); // 잔차 값 배열 채우기
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi); // 동적 공유 초기화
    
    /*** ROS 구독 초기화 ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type[0] == AVIA ? 
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : // LIDAR 타입에 따라 콜백 설정
        nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_pcl2; // 두 번째 LIDAR 콜백 선언
    if (multi_lidar) // 다중 LIDAR 활성화된 경우
    {
        sub_pcl2 = p_pre->lidar_type[1] == AVIA ? 
            nh.subscribe(lid_topic2, 200000, livox_pcl_cbk2) : // 두 번째 LIDAR 콜백 설정
            nh.subscribe(lid_topic2, 200000, standard_pcl_cbk2);
    }
    
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk); // IMU 콜백 설정
    
    // 퍼블리셔 초기화
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000); // 레이저 클라우드 퍼블리셔
    ros::Publisher pubLaserCloudFullTransformed = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_tf", 100000); // 변환된 레이저 클라우드 퍼블리셔
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000); // 바디 프레임 레이저 클라우드 퍼블리셔
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100000); // 레이저 맵 퍼블리셔
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000); // 오도메트리 퍼블리셔
    ros::Publisher pubMavrosVisionPose = nh.advertise<geometry_msgs::PoseStamped>("/mavros/vision_pose/pose", 100000); // MAVROS 비전 포즈 퍼블리셔
    ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/path", 100000); // 경로 퍼블리셔
    ros::Publisher pubCaclTime = nh.advertise<std_msgs::Float32>("/calc_time", 100000); // 계산 시간 퍼블리셔
    ros::Publisher pubPointNum = nh.advertise<std_msgs::Float32>("/point_number", 100000); // 포인트 수 퍼블리셔
    ros::Publisher pubLocalizabilityX = nh.advertise<std_msgs::Float32>("/localizability_x", 100000); // 로컬라이저빌리티 X 퍼블리셔
    ros::Publisher pubLocalizabilityY = nh.advertise<std_msgs::Float32>("/localizability_y", 100000); // 로컬라이저빌리티 Y 퍼블리셔
    ros::Publisher pubLocalizabilityZ = nh.advertise<std_msgs::Float32>("/localizability_z", 100000); // 로컬라이저빌리티 Z 퍼블리셔
    //------------------------------------------------------------------------------------------------------

    signal(SIGINT, SigHandle); // SIGINT 신호 처리 함수 설정
ros::Rate rate(5000); // 루프 주기 설정
bool status = ros::ok(); // ROS 상태 확인

while (status) // ROS가 작동하는 동안 반복
{
    if (flg_exit) break; // 종료 플래그가 설정된 경우 루프 종료
    ros::spinOnce(); // 콜백 함수 호출

    if (sync_packages(Measures)) // 패키지 동기화
    {
        high_resolution_clock::time_point t1 = high_resolution_clock::now(); // 시작 시간 기록
        p_imu->Process(Measures, kf, feats_undistort, multi_lidar); // IMU 처리
        state_point = kf.get_x(); // 상태 추정 가져오기
        pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I; // LIDAR 위치 계산
        if (feats_undistort->empty() || (feats_undistort == NULL))
        {
            ROS_WARN("No point, skip this scan!\n"); // 포인트가 없는 경우 경고 출력
            continue; // 다음 반복으로 이동
        }

        /*** LIDAR FOV 내 맵 세분화 ***/
        lasermap_fov_segment();

        /*** 스캔에서 특징 포인트 다운샘플링 ***/
        downSizeFilterSurf.setInputCloud(feats_undistort); // 입력 포인트 클라우드 설정
        downSizeFilterSurf.filter(*feats_down_body); // 다운샘플링
        feats_down_size = feats_down_body->points.size(); // 다운샘플링된 포인트 수

        /*** 맵 KD 트리 초기화 ***/
        if (ikdtree.Root_Node == nullptr) // KD 트리의 루트 노드가 없는 경우
        {
            if (feats_down_size > 5) // 포인트 수가 5 이상인 경우
            {
                feats_down_world->resize(feats_down_size); // 월드 포인트 클라우드 크기 조정
                for (int i = 0; i < feats_down_size; i++)
                {
                    pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i])); // 포인트 변환
                }                    
                ikdtree.Add_Points(feats_down_world->points, true); // KD 트리에 포인트 추가
            }
            continue; // 다음 반복으로 이동
        }

        /*** ICP 및 반복 칼만 필터 업데이트 ***/
        if (feats_down_size < 5) // 포인트 수가 5 미만인 경우
        {
            ROS_WARN("No point, skip this scan!\n"); // 포인트가 없는 경우 경고 출력
            continue; // 다음 반복으로 이동
        }
        
        normvec->resize(feats_down_size); // 노멀 벡터 크기 조정
        feats_down_world->resize(feats_down_size); // 월드 포인트 클라우드 크기 조정
        Nearest_Points.resize(feats_down_size); // 최근접 포인트 크기 조정
        /*** 반복 상태 추정 ***/
        double solve_H_time = 0; // 자코비안 계산 시간 초기화
        kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time); // 반복 업데이트
        state_point = kf.get_x(); // 상태 추정 가져오기
        pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I; // LIDAR 위치 계산
        geoQuat.x = state_point.rot.coeffs()[0]; // 쿼터니언 설정
        geoQuat.y = state_point.rot.coeffs()[1];
        geoQuat.z = state_point.rot.coeffs()[2];
        geoQuat.w = state_point.rot.coeffs()[3];

        /******* 오도메트리 퍼블리시 *******/
        if (publish_tf_results) publish_visionpose(pubMavrosVisionPose); // 비전 포즈 퍼블리시
        publish_odometry(pubOdomAftMapped); // 오도메리 퍼블리시

        /*** 특징 포인트를 맵 KD 트리에 추가 ***/
        map_incremental();

        if (0) // 맵 포인트를 보고 싶으면 "if(1)"로 변경
        {
            PointVector().swap(ikdtree.PCL_Storage); // KD 트리 저장소 비우기
            ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD); // KD 트리 평탄화
            featsFromMap->clear(); // 맵 포인트 클라우드 초기화
            featsFromMap->points = ikdtree.PCL_Storage; // 맵 포인트 클라우드 설정
        }
        /******* 포인트 퍼블리시 *******/
        if (path_en) publish_path(pubPath); // 경로 퍼블리시
        if (scan_pub_en || pcd_save_en) publish_frame_world(pubLaserCloudFull, pubLaserCloudFullTransformed); // 레이저 클라우드 퍼블리시
        if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body); // 바디 프레임 레이저 클라우드 퍼블리시
        // publish_map(pubLaserCloudMap);
        high_resolution_clock::time_point t2 = high_resolution_clock::now(); // 끝 시간 기록
        auto duration = duration_cast<microseconds>(t2 - t1).count() / 1000.0; // 소요 시간 계산
        std_msgs::Float32 calc_time; 
        calc_time.data = duration; // 계산 시간 설정
        pubCaclTime.publish(calc_time); // 계산 시간 퍼블리시
        std_msgs::Float32 point_num; 
        point_num.data = feats_down_size; // 포인트 수 설정
        pubPointNum.publish(point_num); // 포인트 수 퍼블리시
        std_msgs::Float32 localizability_x, localizability_y, localizability_z; 
        localizability_x.data = localizability_vec(0); // 로컬라이저빌리티 X 설정
        localizability_y.data = localizability_vec(1); // 로컬라이저빌리티 Y 설정
        localizability_z.data = localizability_vec(2); // 로컬라이저빌리티 Z 설정
        pubLocalizabilityX.publish(localizability_x); // 로컬라이저빌리티 X 퍼블리시
        pubLocalizabilityY.publish(localizability_y); // 로컬라이저빌리티 Y 퍼블리시
        pubLocalizabilityZ.publish(localizability_z); // 로컬라이저빌리티 Z 퍼블리시
    }
    status = ros::ok(); // ROS 상태 확인
    rate.sleep(); // 주기 대기
}

/**************** 맵 저장 ****************/
/* 1. 충분한 메모리가 있는지 확인하세요
/* 2. PCD 저장은 실시간 성능에 큰 영향을 미칩니다 **/
if (pcl_wait_save->size() > 0 && pcd_save_en) // PCD 저장 활성화 및 포인트 수 확인
{
    string file_name = string("scans.pcd"); // 파일 이름 설정
    string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name); // 저장 경로 생성
    pcl::PCDWriter pcd_writer; // PCD 작성기 생성
    cout << "current scan saved to /PCD/" << file_name << endl; // 현재 스캔 저장 메시지 출력
    pcd_writer.writeBinary(all_points_dir, *pcl_wait_save); // PCD 파일로 저장
}

return 0; // 프로그램 종료
}
