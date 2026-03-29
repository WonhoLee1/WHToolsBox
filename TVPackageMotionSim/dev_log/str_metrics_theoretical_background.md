# WHTOOLS Structural Analysis Metrics (v4.0) - Theoretical Background

안녕하세요, **WHTOOLS**입니다.
본 문서는 WHTOOLS TVPackageMotionSim v4에 적용된 구조 해석 지표(Structural Metrics)의 공학적 정의, 상세 수식 및 이론적 배경을 기술합니다.
시뮬레이션된 다물체 역학(Multibody Dynamics, MBD) 결과를 기반으로, 유한요소해석(FEA) 수준의 재료 역학적 응력(Stress)과 변형 에너지(Strain Energy)를 역산하여 패키징의 신뢰성을 판단할 수 있는 정량적 지표를 제공합니다.

> [!NOTE]
> 본 모델에 적용된 응력 및 에너지 환산은 Euler-Bernoulli 보 이론과 일반화된 Hooke의 법칙을 기초로 하며, `solref`를 통한 등가 강성 치환 모델을 차용하였습니다.

## 1. 종합 개념 시각화 (Concept Visualization)

![Structural Metrics Overview](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/dev_log/str_metrics_overview_premium.png)

위 그림은 TV 포장 상자 및 완충재(Cushion)가 낙하 충격을 받을 때 발생하는 국부적 응력 집중, 비틀림(Torsion), 주 굽힘 축(Principal Bending Axis)을 시각적으로 나타낸 개념도입니다.

## 2. 국부 지표 (Local Metrics)

개별 요소(Geom/Block) 단위의 극한값을 추적하여, 파손이 가장 먼저 발생할 수 있는 취약 지점을 찾아내는 지표입니다.

### 2.1. 굽힘 응력 및 모멘트 (Bending Stress & Moment)

![Bending Stress](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/dev_log/str_metrics_bs_premium.png)

- **모멘트 ($M$)**: 각 지점에서의 등가 굽힘 모멘트는 회전 강성($K_{rot}$)과 굽힘 각도($\theta$)에 비례합니다.
  $$ M = K_{rot} \cdot \theta = \left( \frac{E_{eff} \cdot I}{L} \right) \cdot \theta $$

- **굽힘 응력 ($\sigma_{bend}$)**: 해당 요소의 단면에 걸리는 최대 압축/인장 응력(MPa)을 산출합니다.
  $$ \sigma_{bend} = \frac{M \cdot c}{I} = \frac{E_{eff} \cdot \theta \cdot (t / 2)}{L} $$
  *(여기서 $t$는 블록의 두께(Thickness) 방향 차원을 의미하여 가장 바깥쪽 파이버 응력을 나타냅니다.)*

### 2.2. 비틀림 응력 (Torsional Stress)

로컬 Z축을 중심으로 한 회전 변형($TA$, Torsion)에 의해 발생하는 전단 응력을 도출합니다.
$$ \tau_{twist} = \frac{T \cdot r}{J} = \frac{(K_{tor} \cdot \theta_{twist}) \cdot r}{J} $$

### 2.3. RRG (Relative Rotation Gradient)

![RRG](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/dev_log/str_metrics_rrg_premium.png)

이웃 블록 간의 각도 구배 수치로, 패널 구조에서 국부적인 '꺾임' 또는 '주름(Wrinkling)' 발생을 경고합니다.
$$ RRG_{i} = \max_{j \in Neighbor(i)} \left( \cos^{-1}\left( \frac{\text{Trace}(R_i^T R_j) - 1}{2} \right) \right) $$

## 3. 전역 지표 (Global Mechanics & Energy)

부품 전체(Component)를 아우르는 통계적·에너지적 거동 지표입니다.

### 3.1. 전단 변형 에너지 (Total Strain Energy, $TSE$)

![Total Strain Energy](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/dev_log/str_metrics_tse_premium.png)

낙하 충격에 의해 완충재/패키지가 시스템적으로 흡수한 운동 에너지와 정적 변형 에너지(Joule)의 총량입니다. 축 방향 압축 및 회전 에너지를 합산합니다.
$$ U_{Total} = \sum_{i=1}^{N} \left( \frac{1}{2} k_{lin,i} (\Delta x_i)^2 + \frac{1}{2} k_{rot,i} (\theta_i)^2 \right) $$

- 완충재의 쿠션 성능과 흡수된 임팩트를 정량적으로 비교할 수 있습니다.

### 3.2. 주 굽힘 축 (Principal Bending Axis, $PBA$)

![Principal Bending Axis](file:///c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/dev_log/str_metrics_pba_premium.png)

Open Cell 패널을 개별 블록의 회전 벡터 집합체로 보고, 이에 대해 공분산 분석(Principal Component Analysis, PCA)을 진행하여 얻습니다. 이는 단순한 X, Y, Z축을 찾는 것이 아니라, **면내에서 회전된 임의의 축(Principal Axis)** 중 가장 지배적인 굽힘 모드가 발생하는 방향과 그 크기를 도출하는 것을 의미합니다.

$$ \mathbf{C} = \frac{1}{N} \sum_{i} \mathbf{u}_i \mathbf{u}_i^T \quad \rightarrow \quad \text{Eigen Decomposition } (\mathbf{C} \mathbf{v} = \lambda \mathbf{v}) $$

- 최대 고유값 $\lambda_{max}$ 에 해당하는 고유벡터 $\mathbf{v}$ 가 해당 시점의 **PBA(주 굽힘 축)** 가 됩니다. 이 축은 패널의 기하학적 축과 무관하게 변형 에너지가 집중되는 실제 물리적 굴곡 축을 나타냅니다.

### 3.3. GTI 및 GBI

- **GTI (Global Tilt Index)**: $ \sqrt{\frac{1}{N} \sum \theta_{tilt,i}^2} $ 구조재 뼈대의 변위량 RMS.
- **GBI (Global Bending Index)**: 부품 전체의 곡률 에너지를 정규화한 값으로 전역 강성을 대변합니다.

## 4. 이론적 참조 (References)

본 WHTOOLS 지표 도출 알고리즘은 아래 공학 서적 및 이론을 기초로 근사/도출되었습니다.

1. Ugural, A. C., & Fenster, S. K. (2011). *Advanced Mechanics of Materials and Applied Elasticity*. Prentice Hall. (Euler-Bernoulli Beam Theory, Torsion)
2. Belytschko, T., Liu, W. K., Moran, B., & Elkhodary, K. (2013). *Nonlinear Finite Elements for Continua and Structures*. John Wiley & Sons. (Strain Energy Density Formulation)
3. Jolliffe, I. T. (2002). *Principal Component Analysis*. Springer. (Basis for Principal Bending Axis - PBA)
