import pickle
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple

@dataclass
class DropSimResult:
    """
    시뮬레이션 전체 결과 데이터를 담는 통합 데이터 구조체입니다.
    이 객체는 바이너리(Pickle) 파일로 저장되어 추후 대량의 DOE 데이터 분석이나 
    실험 데이터와의 매칭(Correlation) 작업에서 핵심 자산으로 활용됩니다.
    
    Attributes:
        config (Dict): 시뮬레이션에 사용된 파라미터 설정값
        metrics (Dict): 컴포넌트별 분석 지표 (Bending, Twist 등)
        max_g_force (float): 시뮬레이션 중 발생한 최대 충격 가속도 (G)
        time_history (List[float]): 시뮬레이션 시간 이력 (s)
        z_hist (List[float]): 낙하 높이(Z축) 이력 (m)
        root_acc_history (List[float]): 중심점 가속도 이력 (m/s^2)
        corner_acc_hist (List[Any]): 8개 모서리의 가속도 이력
        pos_hist (List[Any]): 중심점 위치 이력
        vel_hist (List[Any]): 중심점 속도 이력
        acc_hist (List[Any]): 중심점 6자유도 가속도 이력
        cog_pos_hist (List[Any]): 질량 중심(CoG) 위치 이력
        geo_center_pos_hist (List[Any]): 기하 중심(Geometric Center) 위치 이력
        structural_metrics (Dict): RRG, PBA 등 고급 구조 해석 지표
        critical_timestamps (Dict): 주요 피크 발생 시점 분석 결과
        nominal_local_pos (Dict): 각 블록의 초기 로컬 좌표 매핑 (BodyID -> [x,y,z])
    """
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    max_g_force: float
    time_history: List[float]
    z_hist: List[float]
    root_acc_history: List[float]
    corner_acc_hist: List[Any]
    
    pos_hist: List[Any] = field(default_factory=list)
    vel_hist: List[Any] = field(default_factory=list)
    acc_hist: List[Any] = field(default_factory=list)
    
    cog_pos_hist: List[Any] = field(default_factory=list)
    cog_vel_hist: List[Any] = field(default_factory=list)
    cog_acc_hist: List[Any] = field(default_factory=list)
    
    geo_center_pos_hist: List[Any] = field(default_factory=list)
    geo_center_vel_hist: List[Any] = field(default_factory=list)
    geo_center_acc_hist: List[Any] = field(default_factory=list)
    
    corner_pos_hist: List[Any] = field(default_factory=list)
    corner_vel_hist: List[Any] = field(default_factory=list)
    
    ground_impact_hist: List[float] = field(default_factory=list)
    air_drag_hist: List[float] = field(default_factory=list)
    air_viscous_hist: List[float] = field(default_factory=list)
    air_squeeze_hist: List[float] = field(default_factory=list)
    
    structural_metrics: Dict[str, Any] = field(default_factory=dict)
    critical_timestamps: Dict[str, Any] = field(default_factory=dict)
    nominal_local_pos: Dict[int, List[float]] = field(default_factory=dict)
    
    # [v5.2] 평판 이론(Plate Theory) 분석용 고정밀 데이터 필드
    quat_hist: List[np.ndarray] = field(default_factory=list) # [N_frames, N_bodies, 4]
    components: Dict[str, Dict[Tuple[int, int, int], int]] = field(default_factory=dict) # {PartName: {(i,j,k): BodyID}}
    body_index_map: Dict[int, str] = field(default_factory=dict) # BodyID -> Name
    block_half_extents: Dict[int, List[float]] = field(default_factory=dict) # BodyID -> [dx, dy, dz]
    
    def save(self, filepath: str) -> None:
        """
        시뮬레이션 결과(self)를 지정된 경로에 Pickle 형식으로 저장합니다.
        
        Args:
            filepath (str): 저장할 파일 경로 (.pkl 권장)
        """
        try:
            with open(filepath, "wb") as f:
                pickle.dump(self, f)
        except Exception as e:
            print(f"[DropSimResult] Error saving data: {e}")
            
    @classmethod
    def load(cls, filepath: str) -> 'DropSimResult':
        """
        저장된 바이너리 결과 파일을 읽어 DropSimResult 객체를 복원합니다.
        
        Args:
            filepath (str): 읽어올 Pickle 파일 경로
        Returns:
            DropSimResult: 복원된 데이터 객체
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)
