import gmsh
import math

# =========================================================
# 1. 파라미터 설정
# =========================================================
XX, YY, ZZ = 300.0, 180.0, 5.0

# Hole 치수
a = 20.0    # 세로 길이
b = 5.0     # 가로 길이
c = 5.0     # 세로 간격
d = 10.0    # 가로 간격

# 패턴 영역
PX, PY = 180.0, 180.0

# [옵션] 모서리 라운드 처리 여부 (True: 장공/원형, False: 직사각형)
USE_FILLET = True 

# =========================================================
# 2. 좌표 계산 로직
# =========================================================
def calculate_hole_positions():
    positions = []
    pitch_y = a + c
    pitch_x = b + d
    
    nx = int(math.ceil((PX / 2) / pitch_x)) + 1
    ny = int(math.ceil((PY / 2) / pitch_y)) + 1
    
    for i in range(-nx, nx + 1):
        for j in range(-ny, ny + 1):
            center_x = i * pitch_x
            center_y = j * pitch_y
            if i % 2 != 0: center_y += pitch_y / 2.0
            corner_x = center_x - b/2
            corner_y = center_y - a/2
            
            # X: 포함 조건, Y: 교차 조건
            min_x, max_x = corner_x, corner_x + b
            min_y, max_y = corner_y, corner_y + a
            
            if (min_x >= -PX/2 and max_x <= PX/2) and (min_y < PY/2 and max_y > -PY/2):
                positions.append((corner_x, corner_y))
    return positions

# =========================================================
# 3. GMSH 모델링 (라운드 홀 적용)
# =========================================================
def generate_plate_with_rounded_holes(hole_positions):
    gmsh.initialize()
    gmsh.model.add("plate_rounded_holes")
    factory = gmsh.model.occ

    # 1. Base Plate
    plate = factory.addBox(-XX/2, -YY/2, -ZZ/2, XX, YY, ZZ)
    
    # 2. Mask Box (Y축 트리밍용)
    mask_box = factory.addBox(-PX/2, -PY/2, -ZZ/2 - 2.0, PX, PY, ZZ + 4.0)

    # 3. Holes 생성 (옵션에 따라 분기)
    raw_holes = []
    
    # [수정 포인트 1] -----------------------------------------------------------
    # OpenCASCADE 커널 오류 방지: 
    # Radius가 변의 절반과 정확히 같으면(2.5mm) 직선 구간이 사라져 생성 에러 발생.
    # 따라서 0.001mm 값을 빼주어 안정성을 확보합니다.
    limit_radius = min(a, b) / 2.0
    radius = limit_radius - 0.001  
    if radius < 0: radius = 0 
    # ------------------------------------------------------------------------
    
    # 관통을 위해 Z축 시작점과 길이 설정
    z_start = -ZZ/2 - 1.0
    z_len = ZZ + 2.0
    
    print(f">>> 홀 생성 시작 (Round option: {USE_FILLET}, Radius: {radius:.4f})")

    for (hx, hy) in hole_positions:
        if USE_FILLET:
            # addRectangle(x, y, z, dx, dy, roundedRadius)
            # 이제 radius가 2.499... 이므로 에러가 발생하지 않습니다.
            face_tag = factory.addRectangle(hx, hy, z_start, b, a, roundedRadius=radius)
            
            # Z축으로 압출 (Extrude)
            extrude_out = factory.extrude([(2, face_tag)], 0, 0, z_len)
            
            # extrude 결과에서 volume(dim=3) 태그 찾기
            vol_tag = [tag for dim, tag in extrude_out if dim == 3][0]
            
            raw_holes.append((3, vol_tag))
            
        else:
            # [방식 2] 기존 직육면체 박스
            t = factory.addBox(hx, hy, z_start, b, a, z_len)
            raw_holes.append((3, t))

    if not raw_holes:
        print("생성된 홀이 없습니다.")
        gmsh.finalize()
        return

    # 4. Trimming (Intersection)
    print(f"   - Trimming {len(raw_holes)} holes...")
    
    # [수정 포인트 2] removeTool=False로 설정하여 안전하게 처리 후 수동 삭제
    trimmed_holes, _ = factory.intersect(raw_holes, [(3, mask_box)], removeObject=True, removeTool=False)
    factory.remove([(3, mask_box)])
    
    # 5. Cutting (Difference)
    print("   - Cutting from plate...")
    if trimmed_holes:
        cut_result, _ = factory.cut([(3, plate)], trimmed_holes)
        final_plate_tag = cut_result[0][1]
    else:
        final_plate_tag = plate

    # [옵션] 패턴 영역 라인 스플릿 (Line Split)
    split_curve = factory.addRectangle(-PX/2, -PY/2, ZZ/2, PX, PY)
    fragments, _ = factory.fragment([(3, final_plate_tag)], [(1, split_curve)])

    factory.synchronize()

    # Physical Group 설정
    all_volumes = [tag for dim, tag in fragments if dim == 3]
    gmsh.model.addPhysicalGroup(3, all_volumes, name="ThePlate")

    # 저장
    filename = "plate_with_fillets.step"
    gmsh.write(filename)
    print(f">>> 생성 완료: {filename}")
    
    gmsh.fltk.run()
    gmsh.finalize()

if __name__ == "__main__":
    holes = calculate_hole_positions()
    generate_plate_with_rounded_holes(holes)

    