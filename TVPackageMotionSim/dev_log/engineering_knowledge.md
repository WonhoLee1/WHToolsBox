# WHToolsBox Engineering Knowledge Base

본 문서는 **WHToolsBox** 낙하 시뮬레이션 프레임워크에 적용된 주요 공학적 알고리즘과 물리적 계산 로직을 정리한 기술 가이드입니다.

---

## 1. 구조적 변형 분석 (Structural Distortion Analysis)

이산 블록 모델링에서 조립체의 변형을 정량화하기 위해 상대 회전 행렬(Relative Rotation Matrix) 분해법을 사용합니다.

### 1.1. 상대 회전 행렬 식별
각 블록(Geom)의 글로벌 회전 행렬 $R_{block}$을 조립체 루트(Root)의 회전 행렬 $R_{root}$의 역행렬과 연산하여, 박스 전체의 강체 운동(Rigid Body Motion)을 제거한 **순수 상대 자세** $R_{rel}$을 구합니다.

$$R_{rel} = R_{root}^T \cdot R_{block}$$

이후, 초기 상태의 상대 자세 $R_{rel,0}$를 기준으로 현재의 편차 행렬 $D$를 산출합니다.
$$D = R_{rel,0}^T \cdot R_{rel}$$

### 1.2. 굽힘(Bending) 및 비틂(Twist) 분해
편차 행렬 $D$로부터 공학적으로 유의미한 두 가지 성분을 추출합니다.

- **Bending (Tilt)**: 로컬 Z축이 원래의 법선 방향에서 얼마나 기울어졌는지를 나타냅니다.
  $$\theta_{bend} = \arccos(D_{2,2})$$
- **Twist (Torsion)**: 로컬 Z축을 중심으로 블록이 얼마나 회전했는지를 나타냅니다.
  $$\theta_{twist} = \arctan2(D_{1,0}, D_{0,0})$$

---

## 2. 정밀 공기 역학 (Advanced Aerodynamics)

### 2.1. 항력 (Drag Force)
공기 저항은 일반적인 항력 공식을 사용하되, 박스의 6개 면에 대해 투영 면적을 동적으로 계산하여 적용합니다.
$$F_{drag} = \frac{1}{2} \rho v^2 C_d A$$

### 2.2. 스퀴즈 필름 효과 (Squeeze Film Effect)
지면과 제품 사이의 간극이 좁아질 때 발생하는 압축 공기 쿠션 현상을 모사합니다. 간극 $h$가 작을수록 지수적으로 증가하는 저항력을 부여합니다.

- **압력 모델**: $P_{sq} \propto \frac{\mu V A}{h^3}$ (레이놀즈 방정식의 단순화 모델)
- **구현**: 간극 임계값($h_{max}$) 이하에서 속도 $V$에 비례하고 높이 $h$의 역수에 가중치를 둔 감쇠력을 적용하여 안정적인 착지를 유도합니다.

---

## 3. 소성 변형 알고리즘 (Strain-based Plasticity v3)

단순한 충돌 판정을 넘어, 소재의 강성과 항복점을 고려한 영구 변형 로직입니다.

### 3.1. 듀얼 트리거 시스템 (Dual-Trigger)
소성 변형은 다음 두 가지 조건이 동시에 충족될 때 활성화됩니다.
1. **Strain 조건**: 인접 블록(Neighbor) 간의 거리가 설정된 `yield_strain` 이상 좁아질 때.
2. **Pressure 조건**: 블록 수평 투영 면적 대비 접촉력이 `yield_pressure`를 상과할 때.

### 3.2. 영구 압착 (Permanent Compression)
항복 조건을 만족한 상태에서 하중이 제거(Recovery)되는 시점에, 탄성 복원이 일어나지 않은 만큼의 '기구학적 크기 축소'와 '중심점 이동'을 MuJoCo의 `geom_size`와 `geom_pos`에 실시간으로 반영합니다.

- **Size Reduction**: $S_{new} = S_{old} - \Delta_{plastic} / 2$
- **Position Shift**: $P_{new} = P_{old} \pm \Delta_{plastic} / 2$ (중심부 방향으로 이동)

---

## 4. 시각화 및 데이터 처리

### 4.1. 순위 기반 히트맵 (Rank-based Heatmap)
물리량의 절대값은 부품마다 편차가 크므로, 시각적 대비를 극대화하기 위해 순위 데이터를 사용합니다.
- 점수 $S = (\theta_{bend} + \theta_{twist}) / 2$ 산출
- 부품 내 $N$개 블록에 대해 $S$를 정렬하여 순위 $r$ 부여
- 컬러 팩터 $f = r / (N-1)$를 `RdYlBu_r` 컬러맵에 매핑

---
> [!TIP]
> **WHTOOLS**는 이러한 물리적 근거를 바탕으로 제작되었습니다. 각 파라미터($solref, solimp$)의 조절은 `engineering_knowledge.md`의 수식을 바탕으로 실제 소재의 영률(Young's Modulus) 및 감쇠비와 매칭될 수 있습니다.
