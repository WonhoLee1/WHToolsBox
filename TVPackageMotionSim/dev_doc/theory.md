# WHTOOLS Advanced Physics Modeling: Theoretical Background

본 문서는 **WHTOOLS** TVPackageMotionSim의 핵심 물리 엔진 구동 원리와 구조 해석 알고리즘에 대한 고도화된 이론적 배경을 담고 있습니다. 물리적 정확도(Physical Fidelity)와 연산 효율성(Computational Efficiency)의 균형을 맞추기 위해 도입된 주요 해석 방법론에 대해 서술합니다.

---

## 1. 공력 해석 및 스퀴즈 필름 효과 (Advanced Aerodynamics & Squeeze Film Effect)

자유 낙하 상태의 포장재는 광범위한 표면적(Surface Area)을 가지므로 공기 역학적 저항(Aerodynamic Drag)이 낙하 궤적과 충돌 속도에 지대한 영향을 미칩니다. 본 시뮬레이션 엔진은 고전적인 항력 모델과 함께, 지면 근접 시 발생하는 비선형적 압착 효과를 통합 모델링하여 실제 환경과의 정합성을 극대화합니다.

### 1.1 이차 점성 유동 항력 (Quadratic Aerodynamic Drag)
광범위한 고도에서의 자유 낙하 시 주도적인 감쇠력을 제공하는 고전 유체 역학 모델입니다. 낙하 속도로 인해 전방투영면적(Projected Area)에 가해지는 동압(Dynamic Pressure)에 의해 발생합니다.
* **수식**: $F_{drag} = -\frac{1}{2}\rho C_d A |v_z| v_z$
* **변수 설명**: 여기서 $\rho$는 공기 밀도, $C_d$는 형상에 따른 항력 계수(Drag Coefficient), $A$는 박스의 표면적을 의미합니다. 속도의 제곱에 비례하므로 낙하 속도의 한계치(Terminal Velocity)를 결정짓는 핵심 지표입니다.

### 1.2 선형 점성 마찰 (Linear Viscous Friction)
Reynolds 수가 비교적 낮거나 전단 변형이 지배적인 측면 거동에서 작용하는 선형 감쇠 조건입니다. 난류가 완전히 발달하기 이전 또는 표면 마찰에 의한 미세한 항력을 보강합니다.
* **수식**: $F_{visc} = -\mu C_v A v_z$
* **기능**: $\mu$는 공기의 동점성 계수(Dynamic Viscosity)를 나타냅니다. 속도에 선형 비례하므로 미세한 속도 구간에서의 감쇠(Damping) 안정화에 기여합니다.

### 1.3 스퀴즈 필름 효과 (Squeeze Film Lubrication Effect)
지면 충돌 직전, 포장재의 넓은 바닥면과 지상 사이에 갇힌 얇은 공기층(Air Layer)이 미처 배출되지 못하면서 급격한 압력이 형성되는 현상입니다. Reynolds의 윤활 방정식(Reynolds Lubrication Equation)에 기반하여 지면과 물체 간의 간극($h$)이 극도로 좁아질 때 거시적 감쇠력이 세제곱에 반비례하여 폭발적으로 증가하는 특성을 모사합니다.
* **수식**: $F_{sq} = k_{sq} \mu \frac{A^2 \cdot v_z}{h^3} \quad (\text{for } h < h_{max})$
* **엔지니어링 의의**: 이 모델은 실제 실험 상황에서 포장재가 지면에 도달하기 직전 공기층의 윤활/압착압에 의해 충격량이 부드럽게 감쇠되는(Squeeze damping cushioning) 물리적 현상을 성공적으로 수치해석계로 이식한 것입니다.

---

## 2. 완충재 소성 변형 동역학 (Elastoplastic Constitutive Modeling of Cushions)

EPS(발포폴리스티렌)나 EPP 등 에어셀 기반의 완충 재질은 항복 응력(Yield Stress)을 초과하는 충격하중이 인가될 시 비가역적 파괴 메커니즘(Irreversible Crushing Mechanism)을 거치며 에너지를 영구적으로 소산시킵니다. WHTOOLS 역학 모델은 완전 탄성 충돌의 한계를 극복하기 위해 다축 탄소성(Multi-axial Elastoplasticity) 거동을 실시간으로 추적합니다.

### 2.1 등가 변형률 및 항복 곡면 (Equivalent Strain & Yield Surface)
초기 탄성 구간을 넘어서는 응력이 텐서(Normal/Shear) 형태로 인가되면, 본 구조 해석 엔진은 접촉 수직 벡터(Contact Normal Vector)를 기반으로 다축 상태에서의 등가 변형률(Equivalent Strain, von Mises Criteria의 단순화 변형형)을 산출합니다.
* **접촉 역학 매핑**: $J_2$ 불변량과 유사하게, 물체의 각 요소 블록(Discrete Box Elements)이 경험하는 상대 관입량과 체적 압축량(Volumetric Compression)을 미시적 변형률로 매핑합니다.

### 2.2 에너지 소산 및 소성 경화 루틴 (Energy Dissipation & Strain Hardening)
비선형 동적 외력이 재료의 허용 항복점(Initial Yield Point)을 돌파하면 엔진은 소성 변형률 텐서를 업데이트하여 실시간 복원력(Restoring Force)의 강성(Stiffness) 성분을 영구 무효화(Permanent Deformation)합니다.
* **구현 메커니즘**:
  1. 매 타임스텝 단위 접촉 체적의 수축률 평가
  2. 한계 에너지(Yield Pressure Threshold)를 상회할 시 초과된 에너지를 소성 일(Plastic Work)로 간주해 시스템 소산 에너지 풀(Dissipation Pool)에 귀속시킴
  3. 시각적 피드백: 과도한 소성 변형이 발생한 요소점의 Heatmap Matrix를 업데이트하여 파괴 영역(Crushed Zone)을 가시화

---

## 3. 고정밀 이산-연속체 결합 유연체 모델링 전략 (High-Fidelity Discrete-to-Continuous Flexible Body Modeling Strategy)

대화면 디스플레이(Glass Panel)와 이를 보호하는 얇은 두께의 섀시(Metal Chassis) 복합재의 경우, 단순 강체(Rigid Body) 동역학만으로는 패널의 휨파괴(Bending Failure)나 비틀림(Torsion) 붕괴를 예측할 수 없습니다. 따라서 WHTOOLS는 이산 요소법(DEM, Discrete Element Method)과 표면 복원망 이론을 결합한 독창적 감응 평면 기법을 채택하였습니다.

### 3.1 Reduced-Order Discrete Interconnects (감차원 이산 상호연결)
연속적인 유연 평면을 $N \times M$ 격자망의 강체 유닛 집합체로 분할하고, 각 절점 사이에 점성-탄성 회전-병진 서스펜션(Viscoelastic 6-DOF Joins, `solref`, `solimp`)을 위상수학적으로 바인딩합니다. 이 결합법은 요소의 비선형 대변형(Large Deformation) 구조 방정식을 행렬 역산 없이 빠르고 안정적으로 해석망에 통합시킬 수 있는 이점이 있습니다.

### 3.2 강성 구배 및 곡률 텐서 연산 (Rigidity Gradient & Curvature Tensors)
유연체의 국부적인 변형 징후를 추적하기 위해 인접 블록 간 상대 회전 행렬(Relative Rotation Matrix) $R_{rel}$에서 변환된 회전 자코비안(Rotation Jacobian)을 구합니다.
* **Bending Stress (BS)**: 블록 각 요소 중심 간의 거리 인자($c$), 탄성 계수($E_{eff}$)를 바탕으로 $\sigma_{bend} = \frac{E \cdot \theta \cdot c}{L}$라는 단순화된 Kirchhoff 보 이론을 확장 적용하여 파괴 주응력(Principal Stress) 근사지를 도출합니다.
* **Rotational Rigidity Gradient (RRG)**: 국부적 회전 변화율 $\nabla R = \frac{\partial R}{\partial x}$로 정의되며, 이는 연속체 역학(Continuum Mechanics)에서 응집력의 파계점(Yielding/Tearing points)을 예측하는 지표로 활용됩니다.

### 3.3 연속체 고해상도 재결합 (SSR: Structural Surface Reconstruction)
이산화된 시뮬레이션 결과에서 $C^2$ 계층의 매끄러운 곡면 형상을 복구하여 등고선(Field Contour)을 생성합니다. 이는 **방사형 기저 함수(Radial Basis Function, RBF) 기반의 공간 보간법(Spatial Interpolation)**을 적용하여 이루어집니다. 본 SSR 알고리즘은 거친 입자의 해석 결과에서 정밀 쉘 요소(Shell Element) 기반의 유한요소해석(FEA, Finite Element Analysis)에 준하는 변위/응력 매트릭스를 복원해내어, 실제 충격 시의 박리(Delamination)나 국지적 굴절 패턴을 학술적 정합도(Academic Consistency) 수준으로 표현해 냅니다.
