# Implementation Plan: 변형률(Strain) 기반 소성 변형 로직 (v2) (2026-03-25)

## Goal Description
단순 침투량(Penetration) 대신, `weld`로 연결된 인접 쿠션 블록 간의 **거리 변화(Distance Change)**를 이용한 **변형률(Strain)** 기반 소성 변형 알고리즘을 구현합니다.

## Proposed Changes
- **인접 쌍 탐색**: 코너 블록과 안쪽 블록 간의 인접 정보를 초기화 단계에서 추출.
- **Strain 계산**: `(L_initial - L_current) / L_initial` 공식을 통한 실시간 측정.
- **영구 변형 적용**: 임계 변형률 초과 시 `geom_size`와 `geom_pos` 업데이트.

## Verification
- 로그 감시 및 뷰어 시각화 검증.
