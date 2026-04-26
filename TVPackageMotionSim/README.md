# 🚀 WHTOOLS: Advanced JAX-based FEM Simulation & Multi-PostProcessor

**미분 가능한 물리(Differentiable Physics)와 멀티 에이전트 협업 기반의 차세대 구조 해석 프레임워크**

---

## 🌐 시스템 철학 및 개요
WHTOOLS는 특정 상용 소프트웨어에 종속되지 않는 독립적인 환경에서 **판재(Sheet body) 구조 해석 및 위상/재료 최적화**를 수행하기 위해 설계되었습니다. 단순한 시뮬레이터를 넘어, JAX의 자동 미분(Auto-diff) 기능을 활용하여 물리 현상으로부터 직접 그래디언트를 추출하고 최적의 설계를 도출하는 고성능 엔지니어링 솔루션입니다.

---

## 🤖 멀티 에이전트 아키텍처 (Expert Personas)
본 프로젝트는 각 분야의 전문가 에이전트들이 협업하여 구축되었습니다.
*   **🧠 Agent_FEM**: 쉘(Shell) 및 솔리드 요소 정형화, 비선형 재료(초탄성/점탄성) 구성 방정식 설계.
*   **⚡ Agent_JAX**: `vmap`, `jit` 기반 대규모 병렬 연산 및 민감도 해석 파이프라인 가속화.
*   **🏗️ Agent_Python**: 함수형-OOP 하이브리드 아키텍처 설계 및 독립적 데이터 파이프라인 구축.
*   **📐 Agent_Prof**: 연속체 역학 기반 수식 유도 검증 및 수치 해석 수렴성(Convergence) 분석.
*   **🎨 Agent_Viz**: PyVista 기반의 고성능 3D 렌더링 및 인터랙티브 후처리 스크립트 작성.

---

## 🔥 핵심 기능 상세 (Detailed Features)

### 1. 3D Multi-PostProcessor [V2]
*   **Real-time Deformation Mapping**: 시뮬레이션 결과(Displacement)를 실제 메쉬에 실시간 매핑하여 물리적으로 타당한 변형 형상을 구현.
*   **Advanced Field Visualization**:
    *   주변형률(Principal Strain), 폰 미세스 응력, 평균 곡률(Mean Curvature) 등 텐서/스칼라 필드 지원.
    *   `Scientific Notation`: 지수 표기법 기반의 정밀한 데이터 라벨링.
*   **Interactive ROI Control**:
    *   `Set Rotation Center`: 3D 공간 내 특정 지점을 즉시 회전 중심으로 지정하여 정밀 분석 가능.
    *   `Focus on Part`: 3D 뷰에서 파트 클릭 시 매니저 내 해당 항목으로 즉시 포커싱.
    *   `Perspective/Parallel Toggle`: 분석 용도에 따른 투영 모드 전환.

### 2. Intelligent 2D Analytics Dashboard
*   **Multi-Slot Grid System**: 최대 6개의 슬롯을 지원하며, 각 슬롯은 독립적으로 Contour 또는 Curve(Time-history) 모드로 작동.
*   **Smart Initialization**: 데이터 로딩 시 주변형률 및 곡률 통계를 자동으로 감지하여 최적의 분석 환경을 자율적으로 구성.
*   **Synchronized Navigation**: 3D 애니메이션과 2D 그래프의 시간축이 완벽하게 동기화되어 충격 시점의 물리량 변화를 즉각 포착.

### 3. Dynamic Range & Robustness
*   **Statistical Filtering**: 이상치(Outlier)에 의한 시각적 왜곡을 방지하기 위해 95/98 백분위 기반의 Robust 범위를 자동 산출.
*   **Color LUT Management**: Jet, Viridis, Inferno 등 다양한 컬러맵 지원 및 반전(Reverse) 기능 탑재.

---

## ⚙️ 핵심 알고리즘 및 기술 (Core Algorithms)

### 🛠️ Shell Deformation Analysis
*   마커 기반의 비정형 포인트 클라우드로부터 고차 다항식 근사(Polynomial Approximation)를 통해 연속적인 변형 곡면을 복원.
*   복원된 곡면의 1차/2차 미분 정보를 추출하여 곡률 텐서 및 변형률 에너지 계산.

### ⚡ JAX-accelerated Sensitivity Analysis
*   해석 엔진 자체가 미분 가능하도록 설계되어, 목적 함수(Objective Function)에 대한 설계 변수의 민감도를 자동 미분(`grad`)으로 정확하게 계산.
*   XLA 컴파일러를 통해 CPU/GPU 환경에서 최적화된 기계 코드로 변환되어 대규모 선형 시스템 고속 연산 수행.

---

## 💎 시스템 장점 (Key Advantages)
1.  **압도적 성능**: JAX의 JIT 컴파일을 통해 순수 파이썬 대비 수십 배 이상의 연산 속도 확보.
2.  **높은 유연성**: 외부 라이브러리 의존성을 최소화하여 연구 및 산업 현장의 특수한 요구사항에 맞춰 커스텀 가능.
3.  **전문적 시각화**: 엔지니어링 데이터의 미학적 표현을 위해 다크 모드, 고해상도 그래디언트, 가독성 높은 타이포그래피 적용.
4.  **Zero-Defect 지향**: 엄격한 Null-check와 예외 처리를 통해 장시간 분석 시에도 안정적인 동작 보장.

---

## 🚀 시작하기 (Getting Started)

### 설치 환경
*   Python 3.10 이상
*   주요 의존성: `jax`, `mujoco`, `pyvista`, `pyside6`, `matplotlib`, `koreanize-matplotlib`

### 실행 방법
1.  **시뮬레이션**: `python run_drop_simulation_cases_v6.py` 실행 (결과는 `results/` 폴더에 저장).
2.  **시각화**: `python run_drop_simulation_visualizer.py` 실행하여 최신 결과 자동 로드 및 분석.

---

## 📅 Roadmap & Future Work
*   [ ] 고성능 접촉 마찰 모델(Contact Mechanics) 고도화.
*   [ ] 진화 연산(CMA-ES) 기반의 대규모 설계 공간 탐색 모듈 통합.
*   [ ] 웹 기반 원격 모니터링 대시보드 확장.

---
**Designed & Built by WHTOOLS Advanced Agentic Coding Team.**
