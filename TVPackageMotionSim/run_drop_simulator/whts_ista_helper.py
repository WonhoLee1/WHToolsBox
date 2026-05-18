# -*- coding: utf-8 -*-
"""
[WHTOOLS] ISTA 6A Simulation & Posture Helper Module
===================================================
이 모듈은 box_motion.py에 의존하지 않고 독립적으로 작동하며,
ISTA 6A 규격 판정, 테스트 시퀀스 생성 및 각 시퀀스별 초기 물리 낙하 자세
(Drop Direction, Tilt Latitude/Tilt, Azimuth)를 수치 계산하는 역할을 합니다.

개발자 지침:
- 순수 OOP 구조로 작성되었으며, 각 메서드마다 상세한 Docstring을 포함합니다.
- run_discrete_builder 패키지의 whtb_utils.py/whtb_builder.py의 좌표축 및 방향 키워드와 완벽히 동기화됩니다.
"""

import numpy as np
import re
from typing import Dict, Any, List, Tuple

class IstaFaceMapper:
    """
    ISTA Face Numbering Mapper.
    배송 규격 종류(LTL 여부)에 따라 ISTA 면 번호(1~6)와 기하학적 명칭을 매핑해 줍니다.
    
    Coordinate System (whtb_utils.py 기준):
    - Top: +Y (1), Bottom: -Y (2)
    - Right Side: +X, Left Side: -X
    - Front (Screen): +Z, Back (Rear): -Z
    """
    def __init__(self, is_ltl: bool):
        """
        IstaFaceMapper 초기화.

        Args:
            is_ltl (bool): LTL 화물 운송 여부 (True: LTL, False: Parcel)
        """
        self.is_ltl = is_ltl
        self.mapping = self._build_mapping()
        self.rev_mapping = {v: k for k, v in self.mapping.items()}

    def _build_mapping(self) -> Dict[int, str]:
        """ISTA 면 번호와 기하학적 방향 키의 매핑 딕셔너리를 생성합니다."""
        if self.is_ltl:
            # LTL (Type H/G)
            # 1: Top, 2: Back, 3: Bottom, 4: Front, 5: Right, 6: Left
            return {
                1: 'top',
                2: 'back',
                3: 'bottom',
                4: 'front',
                5: 'right',
                6: 'left'
            }
        else:
            # Parcel (Type A/B/C)
            # 1: Top, 2: Bottom, 3: Right, 4: Left, 5: Front, 6: Back
            return {
                1: 'top',
                2: 'bottom',
                3: 'right',
                4: 'left',
                5: 'front',
                6: 'back'
            }

    def get_face_label(self, geo_face_key: str) -> str:
        """기하학적 면에 대한 ISTA 번호 라벨을 획득합니다 (예: 'Face 1 [Top]')."""
        key = geo_face_key.lower().strip()
        ista_num = self.rev_mapping.get(key, '?')
        long_names = {'top': 'Top', 'bottom': 'Bottom', 'front': 'Front', 'back': 'Back', 'right': 'Right', 'left': 'Left'}
        fname = long_names.get(key, geo_face_key)
        return f"Face {ista_num} ({fname})"

    def get_edge_label(self, f1: str, f2: str) -> str:
        """두 면이 만나 형성하는 엣지에 대한 ISTA 번호 라벨을 획득합니다 (예: 'Edge 3-4 [Bottom-Front]')."""
        k1 = f1.lower().strip()
        k2 = f2.lower().strip()
        n1 = self.rev_mapping.get(k1, '?')
        n2 = self.rev_mapping.get(k2, '?')
        
        # 정렬하여 일관성 유지
        if str(n1) > str(n2):
            n1, n2 = n2, n1
            k1, k2 = k2, k1
            
        long_names = {'top': 'Top', 'bottom': 'Bottom', 'front': 'Front', 'back': 'Back', 'right': 'Right', 'left': 'Left'}
        return f"Edge {n1}-{n2} ({long_names.get(k1, k1)}-{long_names.get(k2, k2)})"

    def get_corner_label(self, f1: str, f2: str, f3: str) -> str:
        """세 면이 만나 형성하는 코너에 대한 ISTA 번호 라벨을 획득합니다 (예: 'Corner 2-3-5 [Bottom-Right-Front]')."""
        keys = [f1.lower().strip(), f2.lower().strip(), f3.lower().strip()]
        nums = [self.rev_mapping.get(k, '?') for k in keys]
        
        # 정렬
        sorted_pairs = sorted(zip(nums, keys), key=lambda x: str(x[0]))
        nums_str = "-".join([str(p[0]) for p in sorted_pairs])
        
        long_names = {'top': 'Top', 'bottom': 'Bottom', 'front': 'Front', 'back': 'Back', 'right': 'Right', 'left': 'Left'}
        faces_str = "-".join([long_names.get(p[1], p[1]) for p in sorted_pairs])
        
        return f"Corner {nums_str} ({faces_str})"


class ISTA6ASimulator:
    """
    ISTA 6A 규격에 근거한 패키지 물리 낙하 시험 조건 산출기.
    치수 및 무게를 토대로 규격 Type(A~H) 판정 및 시퀀스를 생성합니다.
    """
    def __init__(self):
        """ISTA6ASimulator 초기화."""
        pass

    def determine_ista_type(self, mass_kg: float, width_mm: float, depth_mm: float, height_mm: float, 
                           shipment_method: str, handling_method: str, product_type: str = 'General') -> Tuple[str, str]:
        """
        ISTA 6A에 의거하여 규격 Type 및 사유를 판정합니다.

        Args:
            mass_kg (float): 포장물 총 중량 (kg)
            width_mm (float): 가로 크기 (mm)
            depth_mm (float): 두께 크기 (mm)
            height_mm (float): 세로 높이 크기 (mm)
            shipment_method (str): 배송 방식 ('Parcel' 또는 'LTL')
            handling_method (str): 취급 방식 ('Standard' 또는 'Palletized')
            product_type (str): 제품군 ('General' 또는 'TV/Monitor')

        Returns:
            Tuple[str, str]: (규격 코드 A~H/Invalid/Unknown, 판정 사유 문자열)
        """
        dims = sorted([width_mm, depth_mm, height_mm])
        L, W_s, H_s = dims[2], dims[1], dims[0]
        length_plus_girth_mm = L + 2 * (W_s + H_s)
        length_plus_girth_in = length_plus_girth_mm / 25.4
        mass_lb = mass_kg * 2.20462
        
        reason = ""
        type_code = 'Unknown'

        if shipment_method == 'Parcel':
            if length_plus_girth_in > 165.0:
                type_code = 'Invalid'
                reason = f"Parcel selected but Girth+Length {length_plus_girth_in:.1f}in ({length_plus_girth_mm:.0f}mm) > 165in. Must use LTL."
                return type_code, reason
            
            if handling_method == 'Palletized':
                type_code = 'Invalid'
                reason = "Parcel shipment cannot use Palletized handling."
                return type_code, reason

            if mass_lb < 70.0: 
                type_code = 'A'
                reason = f"Parcel (Standard), Weight {mass_lb:.1f}lb ({mass_kg:.1f}kg) < 70lb -> Type A"
            elif mass_lb < 150.0: 
                type_code = 'B'
                reason = f"Parcel (Standard), Weight {mass_lb:.1f}lb ({mass_kg:.1f}kg) 70-150lb -> Type B"
            else: 
                type_code = 'C'
                reason = f"Parcel (Standard), Weight {mass_lb:.1f}lb ({mass_kg:.1f}kg) >= 150lb -> Type C"
            
            reason += f", Girth {length_plus_girth_in:.1f}in ({length_plus_girth_mm:.0f}mm) <= 165in"
            
        elif shipment_method == 'LTL':
            if handling_method == 'Standard':
                if mass_lb < 100.0: 
                    type_code = 'D'
                    reason = f"LTL, Standard, Weight {mass_lb:.1f}lb ({mass_kg:.1f}kg) < 100lb -> Type D"
                else: 
                    type_code = 'E'
                    reason = f"LTL, Standard, Weight {mass_lb:.1f}lb ({mass_kg:.1f}kg) >= 100lb -> Type E"
            
            elif product_type == 'TV/Monitor':
                if mass_lb < 150.0: 
                    type_code = 'G'
                    reason = f"LTL (TV/Monitor), Weight {mass_lb:.1f}lb ({mass_kg:.1f}kg) < 150lb -> Type G"
                else: 
                    type_code = 'H'
                    reason = f"LTL (TV/Monitor), Weight {mass_lb:.1f}lb ({mass_kg:.1f}kg) >= 150lb -> Type H"
            
            else: # Palletized, General
                if mass_lb < 150.0: 
                    type_code = 'F'
                    reason = f"LTL, Palletized, Weight {mass_lb:.1f}lb ({mass_kg:.1f}kg) < 150lb -> Type F"
                else:
                    type_code = 'E'
                    reason = f"LTL, Palletized, Weight {mass_lb:.1f}lb ({mass_kg:.1f}kg) >= 150lb -> Type E"
             
        return type_code, reason

    def generate_test_sequence(self, mass_kg: float, width_mm: float, depth_mm: float, height_mm: float, 
                               shipment_method: str, handling_method: str, product_type: str = 'General') -> Tuple[List[Dict[str, Any]], str]:
        """
        ISTA 규격 판정에 따른 상세 낙하 시험 시퀀스를 생성합니다.

        Args:
            mass_kg (float): 총 중량 (kg)
            width_mm (float): 가로 (mm)
            depth_mm (float): 두께 (mm)
            height_mm (float): 세로 (mm)
            shipment_method (str): 배송 방식
            handling_method (str): 취급 방식
            product_type (str): 제품군

        Returns:
            Tuple[List[Dict[str, Any]], str]: (시퀀스 스텝 딕셔너리 리스트, 규격 코드)
        """
        type_code, _ = self.determine_ista_type(mass_kg, width_mm, depth_mm, height_mm, shipment_method, handling_method, product_type)
        
        is_ltl = type_code in ['D', 'E', 'F', 'G', 'H']
        mapper = IstaFaceMapper(is_ltl=is_ltl)
        
        seq = []
        mass_lb = mass_kg * 2.20462

        if not is_ltl:
            # Parcel (Type A, B, C) - 17 Steps (Block 2 & 15)
            if mass_lb < 70.0:
                h_std = 460.0    # Standard drop: 460mm (18")
                h_high = 910.0   # High drop: 910mm (36")
            else:
                h_std = 300.0    # Standard drop: 300mm (12")
                h_high = 610.0   # High drop: 610mm (24")

            # 1. Edge Top-Front
            seq.append({
                'num': 1, 'type': 'edge', 'direction': 'front-top',
                'name': mapper.get_edge_label('front', 'top'),
                'height': h_std, 'tilt_lat': 0, 'tilt_az': 0,
                'desc': f"1. {mapper.get_edge_label('front', 'top')} - {h_std:.0f}mm"
            })
            # 2. Edge Front-Left
            seq.append({
                'num': 2, 'type': 'edge', 'direction': 'front-left',
                'name': mapper.get_edge_label('front', 'left'),
                'height': h_std, 'tilt_lat': 0, 'tilt_az': 0,
                'desc': f"2. {mapper.get_edge_label('front', 'left')} - {h_std:.0f}mm"
            })
            # 3. Edge Top-Left
            seq.append({
                'num': 3, 'type': 'edge', 'direction': 'top-left',
                'name': mapper.get_edge_label('top', 'left'),
                'height': h_std, 'tilt_lat': 0, 'tilt_az': 0,
                'desc': f"3. {mapper.get_edge_label('top', 'left')} - {h_std:.0f}mm"
            })
            # 4. Corner Front-Top-Left
            seq.append({
                'num': 4, 'type': 'corner', 'direction': 'front-top-left',
                'name': mapper.get_corner_label('front', 'top', 'left'),
                'height': h_std, 'tilt_lat': 0, 'tilt_az': 0,
                'desc': f"4. {mapper.get_corner_label('front', 'top', 'left')} - {h_std:.0f}mm"
            })
            # 5. Corner Bottom-Front-Right
            seq.append({
                'num': 5, 'type': 'corner', 'direction': 'front-bottom-right',
                'name': mapper.get_corner_label('front', 'bottom', 'right'),
                'height': h_std, 'tilt_lat': 0, 'tilt_az': 0,
                'desc': f"5. {mapper.get_corner_label('front', 'bottom', 'right')} - {h_std:.0f}mm"
            })
            # 6. Edge Bottom-Front
            seq.append({
                'num': 6, 'type': 'edge', 'direction': 'front-bottom',
                'name': mapper.get_edge_label('front', 'bottom'),
                'height': h_std, 'tilt_lat': 0, 'tilt_az': 0,
                'desc': f"6. {mapper.get_edge_label('front', 'bottom')} - {h_std:.0f}mm"
            })
            # 7. Edge Back-Bottom
            seq.append({
                'num': 7, 'type': 'edge', 'direction': 'back-bottom',
                'name': mapper.get_edge_label('back', 'bottom'),
                'height': h_std, 'tilt_lat': 0, 'tilt_az': 0,
                'desc': f"7. {mapper.get_edge_label('back', 'bottom')} - {h_std:.0f}mm"
            })
            # 8. Face Front [High]
            seq.append({
                'num': 8, 'type': 'face', 'direction': 'front',
                'name': mapper.get_face_label('front') + " [High]",
                'height': h_high, 'tilt_lat': 0, 'tilt_az': 0,
                'desc': f"8. {mapper.get_face_label('front')} [High] - {h_high:.0f}mm"
            })
            # 9. Face Front [Low]
            seq.append({
                'num': 9, 'type': 'face', 'direction': 'front',
                'name': mapper.get_face_label('front') + " [Low]",
                'height': h_std, 'tilt_lat': 0, 'tilt_az': 0,
                'desc': f"9. {mapper.get_face_label('front')} [Low] - {h_std:.0f}mm"
            })
            # 10. Edge Top-Front (Repeated)
            seq.append({
                'num': 10, 'type': 'edge', 'direction': 'front-top',
                'name': mapper.get_edge_label('front', 'top') + " (Repeat)",
                'height': h_std, 'tilt_lat': 0, 'tilt_az': 0,
                'desc': f"10. {mapper.get_edge_label('front', 'top')} - {h_std:.0f}mm"
            })
            # 11. Edge Front-Left (Repeated)
            seq.append({
                'num': 11, 'type': 'edge', 'direction': 'front-left',
                'name': mapper.get_edge_label('front', 'left') + " (Repeat)",
                'height': h_std, 'tilt_lat': 0, 'tilt_az': 0,
                'desc': f"11. {mapper.get_edge_label('front', 'left')} - {h_std:.0f}mm"
            })
            # 12. Edge Back-Right
            seq.append({
                'num': 12, 'type': 'edge', 'direction': 'back-right',
                'name': mapper.get_edge_label('back', 'right'),
                'height': h_std, 'tilt_lat': 0, 'tilt_az': 0,
                'desc': f"12. {mapper.get_edge_label('back', 'right')} - {h_std:.0f}mm"
            })
            # 13. Corner Front-Top-Left (Repeated)
            seq.append({
                'num': 13, 'type': 'corner', 'direction': 'front-top-left',
                'name': mapper.get_corner_label('front', 'top', 'left') + " (Repeat)",
                'height': h_std, 'tilt_lat': 0, 'tilt_az': 0,
                'desc': f"13. {mapper.get_corner_label('front', 'top', 'left')} - {h_std:.0f}mm"
            })
            # 14. Corner Back-Bottom-Left
            seq.append({
                'num': 14, 'type': 'corner', 'direction': 'back-bottom-left',
                'name': mapper.get_corner_label('back', 'bottom', 'left'),
                'height': h_std, 'tilt_lat': 0, 'tilt_az': 0,
                'desc': f"14. {mapper.get_corner_label('back', 'bottom', 'left')} - {h_std:.0f}mm"
            })
            # 15. Corner Back-Top-Right
            seq.append({
                'num': 15, 'type': 'corner', 'direction': 'back-top-right',
                'name': mapper.get_corner_label('back', 'top', 'right'),
                'height': h_std, 'tilt_lat': 0, 'tilt_az': 0,
                'desc': f"15. {mapper.get_corner_label('back', 'top', 'right')} - {h_std:.0f}mm"
            })
            # 16. Face Left [High]
            seq.append({
                'num': 16, 'type': 'face', 'direction': 'left',
                'name': mapper.get_face_label('left') + " [High]",
                'height': h_high, 'tilt_lat': 0, 'tilt_az': 0,
                'desc': f"16. {mapper.get_face_label('left')} [High] - {h_high:.0f}mm"
            })
            # 17. Hazard Drop (Screen Face 3)
            seq.append({
                'num': 17, 'type': 'face', 'direction': 'front',
                'name': "Hazard Drop " + mapper.get_face_label('front'),
                'height': h_std, 'tilt_lat': 0, 'tilt_az': 0,
                'desc': f"17. Hazard Drop on {mapper.get_face_label('front')} - {h_std:.0f}mm"
            })

        else:
            # LTL (Type D, E, F, G, H)
            # Rotational 혹은 일반 Drop 판별
            is_heavy_ltl = (mass_lb >= 100.0) or (type_code in ['G', 'H'])

            if is_heavy_ltl:
                # Rotational Drop Sequence - Pivot 및 회전 경사 각도(Tilt) 자동 산출
                lift_h = 230.0 # 9 inches = 230mm
                
                # 1. Rot Edge Back-Bottom
                arm_d = depth_mm / 1000.0 # m 단위 변환
                lat_d = np.degrees(np.arcsin(min(1.0, (lift_h / 1000.0) / arm_d))) if arm_d > 0 else 0.0
                seq.append({
                    'num': 1, 'type': 'rot_edge', 'direction': 'back-bottom',
                    'name': 'Rot. Edge: Pivot Back-Bottom',
                    'height': lift_h / 1000.0, 'tilt_lat': lat_d, 'tilt_az': 0,
                    'desc': f"1. Rot. Edge: Pivot Back-Bottom - Lift {lift_h:.0f}mm (Tilt: {lat_d:.1f}°)"
                })
                # 2. Rot Edge Front-Bottom
                seq.append({
                    'num': 2, 'type': 'rot_edge', 'direction': 'front-bottom',
                    'name': 'Rot. Edge: Pivot Front-Bottom',
                    'height': lift_h / 1000.0, 'tilt_lat': lat_d, 'tilt_az': 0,
                    'desc': f"2. Rot. Edge: Pivot Front-Bottom - Lift {lift_h:.0f}mm (Tilt: {lat_d:.1f}°)"
                })
                # 3. Rot Edge Right-Bottom
                arm_w = width_mm / 1000.0
                lat_w = np.degrees(np.arcsin(min(1.0, (lift_h / 1000.0) / arm_w))) if arm_w > 0 else 0.0
                seq.append({
                    'num': 3, 'type': 'rot_edge', 'direction': 'right-bottom',
                    'name': 'Rot. Edge: Pivot Right-Bottom',
                    'height': lift_h / 1000.0, 'tilt_lat': lat_w, 'tilt_az': 0,
                    'desc': f"3. Rot. Edge: Pivot Right-Bottom - Lift {lift_h:.0f}mm (Tilt: {lat_w:.1f}°)"
                })
                # 4. Rot Edge Left-Bottom
                seq.append({
                    'num': 4, 'type': 'rot_edge', 'direction': 'left-bottom',
                    'name': 'Rot. Edge: Pivot Left-Bottom',
                    'height': lift_h / 1000.0, 'tilt_lat': lat_w, 'tilt_az': 0,
                    'desc': f"4. Rot. Edge: Pivot Left-Bottom - Lift {lift_h:.0f}mm (Tilt: {lat_w:.1f}°)"
                })
                # 5. Rot Corner Back-Left-Bottom
                arm_diag = np.sqrt(width_mm**2 + depth_mm**2) / 1000.0
                lat_diag = np.degrees(np.arcsin(min(1.0, (lift_h / 1000.0) / arm_diag))) if arm_diag > 0 else 0.0
                seq.append({
                    'num': 5, 'type': 'rot_corner', 'direction': 'back-left-bottom',
                    'name': 'Rot. Corner: Pivot Back-Left-Bottom',
                    'height': lift_h / 1000.0, 'tilt_lat': lat_diag, 'tilt_az': 0,
                    'desc': f"5. Rot. Corner: Pivot Back-Left-Bottom - Lift {lift_h:.0f}mm (Tilt: {lat_diag:.1f}°)"
                })

            else:
                # LTL Lightweight (<100lb) - 12 Drops (Block 4 & 16)
                h_12 = 300.0  # 12"
                h_18 = 460.0  # 18"
                h_32 = 810.0  # 32"

                # 1. Face 1 (Top)
                seq.append({
                    'num': 1, 'type': 'face', 'direction': 'top',
                    'name': mapper.get_face_label('top'),
                    'height': h_12, 'tilt_lat': 0, 'tilt_az': 0,
                    'desc': f"1. {mapper.get_face_label('top')} - {h_12:.0f}mm"
                })
                # 2. Face 2 (Back)
                seq.append({
                    'num': 2, 'type': 'face', 'direction': 'back',
                    'name': mapper.get_face_label('back'),
                    'height': h_12, 'tilt_lat': 0, 'tilt_az': 0,
                    'desc': f"2. {mapper.get_face_label('back')} - {h_12:.0f}mm"
                })
                # 3. Face 6 (Left)
                seq.append({
                    'num': 3, 'type': 'face', 'direction': 'left',
                    'name': mapper.get_face_label('left'),
                    'height': h_12, 'tilt_lat': 0, 'tilt_az': 0,
                    'desc': f"3. {mapper.get_face_label('left')} - {h_12:.0f}mm"
                })
                # 4. Corner 2-3-5 (Back-Bottom-Right)
                seq.append({
                    'num': 4, 'type': 'corner', 'direction': 'back-bottom-right',
                    'name': mapper.get_corner_label('back', 'bottom', 'right'),
                    'height': h_12, 'tilt_lat': 0, 'tilt_az': 0,
                    'desc': f"4. {mapper.get_corner_label('back', 'bottom', 'right')} - {h_12:.0f}mm"
                })
                # 5. Edge 3-4 (Bottom-Front)
                seq.append({
                    'num': 5, 'type': 'edge', 'direction': 'front-bottom',
                    'name': mapper.get_edge_label('front', 'bottom'),
                    'height': h_12, 'tilt_lat': 0, 'tilt_az': 0,
                    'desc': f"5. {mapper.get_edge_label('front', 'bottom')} - {h_12:.0f}mm"
                })
                # 6. Face 3 (Bottom)
                seq.append({
                    'num': 6, 'type': 'face', 'direction': 'bottom',
                    'name': mapper.get_face_label('bottom'),
                    'height': h_18, 'tilt_lat': 0, 'tilt_az': 0,
                    'desc': f"6. {mapper.get_face_label('bottom')} - {h_18:.0f}mm"
                })
                # 7. Edge 2-3 (Back-Bottom)
                seq.append({
                    'num': 7, 'type': 'edge', 'direction': 'back-bottom',
                    'name': mapper.get_edge_label('back', 'bottom'),
                    'height': h_18, 'tilt_lat': 0, 'tilt_az': 0,
                    'desc': f"7. {mapper.get_edge_label('back', 'bottom')} - {h_18:.0f}mm"
                })
                # 8. Corner 3-4-6 (Bottom-Front-Left)
                seq.append({
                    'num': 8, 'type': 'corner', 'direction': 'front-bottom-left',
                    'name': mapper.get_corner_label('front', 'bottom', 'left'),
                    'height': h_18, 'tilt_lat': 0, 'tilt_az': 0,
                    'desc': f"8. {mapper.get_corner_label('front', 'bottom', 'left')} - {h_18:.0f}mm"
                })
                # 9. Edge 4-5 (Front-Right)
                seq.append({
                    'num': 9, 'type': 'edge', 'direction': 'front-right',
                    'name': mapper.get_edge_label('front', 'right'),
                    'height': h_18, 'tilt_lat': 0, 'tilt_az': 0,
                    'desc': f"9. {mapper.get_edge_label('front', 'right')} - {h_18:.0f}mm"
                })
                # 10. Corner 1-4-6 (Top-Front-Left)
                seq.append({
                    'num': 10, 'type': 'corner', 'direction': 'front-top-left',
                    'name': mapper.get_corner_label('front', 'top', 'left'),
                    'height': h_18, 'tilt_lat': 0, 'tilt_az': 0,
                    'desc': f"10. {mapper.get_corner_label('front', 'top', 'left')} - {h_18:.0f}mm"
                })
                # 11. Edge 1-6 (Top-Left)
                seq.append({
                    'num': 11, 'type': 'edge', 'direction': 'top-left',
                    'name': mapper.get_edge_label('top', 'left'),
                    'height': h_18, 'tilt_lat': 0, 'tilt_az': 0,
                    'desc': f"11. {mapper.get_edge_label('top', 'left')} - {h_18:.0f}mm"
                })
                # 12. Face 3 (Bottom) [High]
                seq.append({
                    'num': 12, 'type': 'face', 'direction': 'bottom',
                    'name': mapper.get_face_label('bottom') + " [High]",
                    'height': h_32, 'tilt_lat': 0, 'tilt_az': 0,
                    'desc': f"12. {mapper.get_face_label('bottom')} [High] - {h_32:.0f}mm"
                })

        return seq, type_code
