#include "preprocess.h" // preprocess.h 헤더 파일 포함

#define RETURN0     0x00 // RETURN0 매크로 정의
#define RETURN0AND1 0x10 // RETURN0AND1 매크로 정의

Preprocess::Preprocess() // Preprocess 생성자
{
  lidar_type[0] = 1; // 첫 번째 LiDAR 타입 초기화
  lidar_type[1] = 1; // 두 번째 LiDAR 타입 초기화
  point_filter_num[0] = 3; // 첫 번째 포인트 필터 수 초기화
  point_filter_num[1] = 3; // 두 번째 포인트 필터 수 초기화
  N_SCANS[0] = 6; // 첫 번째 스캔 수 초기화
  N_SCANS[1] = 6; // 두 번째 스캔 수 초기화
  SCAN_RATE[0] = 10; // 첫 번째 스캔 속도 초기화
  SCAN_RATE[1] = 10; // 두 번째 스캔 속도 초기화
  time_unit[0] = 2; // 첫 번째 시간 단위 초기화
  time_unit[1] = 2; // 두 번째 시간 단위 초기화
  blind[0] = 0.1; // 첫 번째 블라인드 값 초기화
  blind[1] = 0.1; // 두 번째 블라인드 값 초기화
  given_offset_time = false; // 오프셋 시간이 주어졌는지 여부 초기화
}

Preprocess::~Preprocess() {} // Preprocess 소멸자

void Preprocess::set() // 설정 함수
{
  switch (time_unit[0]) // 첫 번째 시간 단위에 따라
  {
    case SEC: // 초 단위인 경우
      time_unit_scale[0] = 1.e3f; // 시간 단위 스케일 설정
      break;
    case MS: // 밀리초 단위인 경우
      time_unit_scale[0] = 1.f; // 시간 단위 스케일 설정
      break;
    case US: // 마이크로초 단위인 경우
      time_unit_scale[0] = 1.e-3f; // 시간 단위 스케일 설정
      break;
    case NS: // 나노초 단위인 경우
      time_unit_scale[0] = 1.e-6f; // 시간 단위 스케일 설정
      break;
    default: // 기본값
      time_unit_scale[0] = 1.f; // 시간 단위 스케일 설정
      break;
  }
  switch (time_unit[1]) // 두 번째 시간 단위에 따라
  {
    case SEC: // 초 단위인 경우
      time_unit_scale[1] = 1.e3f; // 시간 단위 스케일 설정
      break;
    case MS: // 밀리초 단위인 경우
      time_unit_scale[1] = 1.f; // 시간 단위 스케일 설정
      break;
    case US: // 마이크로초 단위인 경우
      time_unit_scale[1] = 1.e-3f; // 시간 단위 스케일 설정
      break;
    case NS: // 나노초 단위인 경우
      time_unit_scale[1] = 1.e-6f; // 시간 단위 스케일 설정
      break;
    default: // 기본값
      time_unit_scale[1] = 1.f; // 시간 단위 스케일 설정
      break;
  }
  return; // 함수 종료
}

void Preprocess::process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out, const int &lidar_num)
{  
    avia_handler(msg, lidar_num); // AVIA LiDAR 메시지를 처리하는 함수 호출
    *pcl_out = pl_surf; // 처리된 포인트 클라우드를 출력 포인터에 할당
}

void Preprocess::process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out, const int &lidar_num)
{
    switch (lidar_type[lidar_num]) // LiDAR 타입에 따라 처리
    {
    case OUST64: // OUST64 타입인 경우
        oust64_handler(msg, lidar_num); // OUST64 LiDAR 메시지를 처리하는 함수 호출
        break;

    case VELO16: // VELO16 타입인 경우
        velodyne_handler(msg, lidar_num); // VELO16 LiDAR 메시지를 처리하는 함수 호출
        break;
  
    default: // 정의되지 않은 LiDAR 타입인 경우
        printf("Error LiDAR Type"); // 오류 메시지 출력
        break;
    }
    *pcl_out = pl_surf; // 처리된 포인트 클라우드를 출력 포인터에 할당
}
void Preprocess::avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg, const int &lidar_num)
{
    pl_surf.clear(); // 표면 포인트 클라우드 초기화
    pl_full.clear(); // 전체 포인트 클라우드 초기화
    double t1 = omp_get_wtime(); // 현재 시간 기록
    int plsize = msg->point_num; // 포인트 수 가져오기
    // cout<<"plsize: "<<plsize<<endl; // 포인트 수 출력 (주석 처리됨)

    pl_surf.reserve(plsize); // 표면 포인트 클라우드의 용량 예약
    pl_full.resize(plsize); // 전체 포인트 클라우드 크기 조정

    for(int i=0; i<N_SCANS[lidar_num]; i++) // 각 스캔에 대해 반복
    {
        pl_buff[i].clear(); // 스캔 버퍼 초기화
        pl_buff[i].reserve(plsize); // 스캔 버퍼의 용량 예약
    }
    uint valid_num = 0; // 유효 포인트 수 초기화
    
    for(uint i=1; i<plsize; i++) // 포인트 수만큼 반복
    {
        // 유효한 포인트인지 확인
        if((msg->points[i].line < N_SCANS[lidar_num]) && ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00))
        {
            valid_num++; // 유효 포인트 수 증가
            if (valid_num % point_filter_num[lidar_num] == 0) // 포인트 필터 수에 따라 필터링
            {
                pl_full[i].x = msg->points[i].x; // X 좌표 설정
                pl_full[i].y = msg->points[i].y; // Y 좌표 설정
                pl_full[i].z = msg->points[i].z; // Z 좌표 설정
                pl_full[i].intensity = msg->points[i].reflectivity; // 강도 설정
                pl_full[i].curvature = msg->points[i].offset_time / float(1000000); // 각 레이저 포인트의 시간으로 곡률 설정 (단위: ms)

                // 이전 포인트와의 거리 및 블라인드 값을 기반으로 표면 포인트 추가
                if(((abs(pl_full[i].x - pl_full[i-1].x) > 1e-7) || 
                    (abs(pl_full[i].y - pl_full[i-1].y) > 1e-7) || 
                    (abs(pl_full[i].z - pl_full[i-1].z) > 1e-7)) &&
                   (pl_full[i].x * pl_full[i].x + pl_full[i].y * pl_full[i].y + pl_full[i].z * pl_full[i].z > 
                   (blind[lidar_num] * blind[lidar_num])))
                {
                    pl_surf.push_back(pl_full[i]); // 유효한 포인트를 표면 포인트 클라우드에 추가
                }
            }
        }
    }
}


void Preprocess::oust64_handler(const sensor_msgs::PointCloud2::ConstPtr &msg, const int &lidar_num)
{
    pl_surf.clear(); // 표면 포인트 클라우드 초기화
    pl_full.clear(); // 전체 포인트 클라우드 초기화
    pcl::PointCloud<ouster_ros::Point> pl_orig; // 원본 포인트 클라우드 선언
    pcl::fromROSMsg(*msg, pl_orig); // ROS 메시지를 PCL 포인트 클라우드로 변환
    int plsize = pl_orig.size(); // 원본 포인트 클라우드 크기 가져오기
    pl_surf.reserve(plsize); // 표면 포인트 클라우드의 용량 예약
    double time_stamp = msg->header.stamp.toSec(); // 타임스탬프 가져오기

    for (int i = 0; i < pl_orig.points.size(); i++) // 모든 포인트에 대해 반복
    {
        if (i % point_filter_num[lidar_num] != 0) continue; // 포인트 필터에 따라 건너뛰기

        double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y + pl_orig.points[i].z * pl_orig.points[i].z; // 포인트 거리 계산
        
        if (range < (blind[lidar_num] * blind[lidar_num])) continue; // 블라인드 값보다 작은 경우 건너뛰기
        
        Eigen::Vector3d pt_vec; // 포인트 벡터 선언
        PointType added_pt; // 추가할 포인트 타입 선언
        added_pt.x = pl_orig.points[i].x; // X 좌표 설정
        added_pt.y = pl_orig.points[i].y; // Y 좌표 설정
        added_pt.z = pl_orig.points[i].z; // Z 좌표 설정
        added_pt.intensity = pl_orig.points[i].intensity; // 강도 설정
        added_pt.normal_x = 0; // 법선 벡터 X 초기화
        added_pt.normal_y = 0; // 법선 벡터 Y 초기화
        added_pt.normal_z = 0; // 법선 벡터 Z 초기화
        added_pt.curvature = pl_orig.points[i].t * time_unit_scale[lidar_num]; // 곡률을 타임스탬프 값으로 설정 (단위: ms)

        pl_surf.points.push_back(added_pt); // 표면 포인트 클라우드에 포인트 추가
    }
}

void Preprocess::velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg, const int &lidar_num)
{
    pl_surf.clear(); // 표면 포인트 클라우드 초기화
    pl_full.clear(); // 전체 포인트 클라우드 초기화

    pcl::PointCloud<velodyne_ros::Point> pl_orig; // 원본 포인트 클라우드 선언
    pcl::fromROSMsg(*msg, pl_orig); // ROS 메시지를 PCL 포인트 클라우드로 변환
    int plsize = pl_orig.points.size(); // 원본 포인트 클라우드 크기 가져오기
    if (plsize == 0) return; // 포인트가 없으면 함수 종료
    pl_surf.reserve(plsize); // 표면 포인트 클라우드의 용량 예약

    /*** 포인트 타임스탬프가 주어지지 않은 경우에만 사용하는 변수 ***/
    double omega_l = 0.361 * SCAN_RATE[lidar_num]; // 스캔 각속도 계산
    std::vector<bool> is_first(N_SCANS[lidar_num], true); // 각 스캔 레이어의 첫 번째 포인트 여부
    std::vector<double> yaw_fp(N_SCANS[lidar_num], 0.0); // 첫 번째 스캔 포인트의 요 각도
    std::vector<float> yaw_last(N_SCANS[lidar_num], 0.0); // 마지막 스캔 포인트의 요 각도
    std::vector<float> time_last(N_SCANS[lidar_num], 0.0); // 마지막 오프셋 시간
    /*****************************************************************/

    if (pl_orig.points[plsize - 1].time > 0) // 마지막 포인트의 타임스탬프가 주어진 경우
    {
        given_offset_time = true; // 오프셋 시간이 주어졌다고 설정
    }
    else // 타임스탬프가 주어지지 않은 경우
    {
        given_offset_time = false; // 오프셋 시간이 주어지지 않았다고 설정
        double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578; // 첫 번째 포인트의 요 각도 계산
        double yaw_end = yaw_first; // 마지막 요 각도 초기화
        int layer_first = pl_orig.points[0].ring; // 첫 번째 포인트의 레이어 저장
        for (uint i = plsize - 1; i > 0; i--) // 마지막 포인트부터 첫 번째 포인트까지 반복
        {
            if (pl_orig.points[i].ring == layer_first) // 같은 레이어인 경우
            {
                yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578; // 마지막 포인트의 요 각도 계산
                break; // 반복 종료
            }
        }
    }

    for (int i = 0; i < plsize; i++) // 모든 포인트에 대해 반복
    {
        // if ( (abs(pl_orig.points[i].time) < 1.0 / SCAN_RATE[lidar_num] / 1800) || (abs(pl_orig.points[i].time) > (1.0 / SCAN_RATE[lidar_num]) * 1.1) )
        // {
        //   continue; // 특정 조건을 만족하지 않으면 건너뛰기 (주석 처리됨)
        // }
        PointType added_pt; // 추가할 포인트 타입 선언
        // cout<<"!!!!!!"<<i<<" "<<plsize<<endl; // 현재 포인트 인덱스 출력 (주석 처리됨)

        added_pt.normal_x = 0; // 법선 벡터 X 초기화
        added_pt.normal_y = 0; // 법선 벡터 Y 초기화
        added_pt.normal_z = 0; // 법선 벡터 Z 초기화
        added_pt.x = pl_orig.points[i].x; // X 좌표 설정
        added_pt.y = pl_orig.points[i].y; // Y 좌표 설정
        added_pt.z = pl_orig.points[i].z; // Z 좌표 설정
        added_pt.intensity = pl_orig.points[i].intensity; // 강도 설정
        added_pt.curvature = pl_orig.points[i].time * time_unit_scale[lidar_num]; // 곡률을 타임스탬프 값으로 설정 (단위: ms)

        if (!given_offset_time) // 오프셋 시간이 주어지지 않은 경우
        {
            int layer = pl_orig.points[i].ring; // 현재 포인트의 레이어 저장
            double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957; // 현재 포인트의 요 각도 계산

            if (is_first[layer]) // 현재 레이어의 첫 번째 포인트인 경우
            {
                yaw_fp[layer] = yaw_angle; // 첫 번째 포인트의 요 각도 저장
                is_first[layer] = false; // 첫 번째 포인트 여부 변경
                added_pt.curvature = 0.0; // 곡률 초기화
                yaw_last[layer] = yaw_angle; // 마지막 요 각도 저장
                time_last[layer] = added_pt.curvature; // 마지막 오프셋 시간 저장
                continue; // 다음 반복으로 넘어감
            }

            // 오프셋 시간 계산
            if (yaw_angle <= yaw_fp[layer]) // 현재 요 각도가 첫 번째 포인트의 요 각도보다 작거나 같은 경우
            {
                added_pt.curvature = (yaw_fp[layer] - yaw_angle) / omega_l; // 오프셋 시간 계산
            }
            else // 현재 요 각도가 첫 번째 포인트의 요 각도보다 큰 경우
            {
                added_pt.curvature = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l; // 오프셋 시간 계산
            }

            if (added_pt.curvature < time_last[layer]) // 마지막 오프셋 시간보다 작으면
                added_pt.curvature += 360.0 / omega_l; // 360도 보정

            yaw_last[layer] = yaw_angle; // 마지막 요 각도 저장
            time_last[layer] = added_pt.curvature; // 마지막 오프셋 시간 저장
        }

        if (i % point_filter_num[lidar_num] == 0) // 포인트 필터 수에 따라 필터링
        {
            if (added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z > (blind[lidar_num] * blind[lidar_num])) // 블라인드 값보다 큰 경우
            {
                pl_surf.points.push_back(added_pt); // 표면 포인트 클라우드에 포인트 추가
            }
        }
    }
}


