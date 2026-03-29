# MCP (Model Context Protocol) 오류 해결 계획

안녕하세요, **WHTOOLS**입니다.

사용자께서 문의하신 **MCP(Model Context Protocol)** 관련 오류는 최근 엔지니어들 사이에서 가장 화두가 되고 있는 인터페이스 기술인 만큼, 초기 설정이나 특정 환경(특히 Windows)에서 의도치 않은 작동 불능 현상이 잦은 편입니다.

본 계획서에서는 일반적인 MCP 오류의 원인을 분석하고, 이를 체계적으로 해결하기 위한 로드맵을 제안합니다.

---

## 1. 개요 및 배경
MCP는 LLM 컨텍스트를 확장하기 위한 강력한 도구이나, 클라이언트(Claude Desktop 등)와 서버(Local scripts) 간의 정밀한 통신 프로토콜을 준수해야 합니다. "MCP error"는 주로 이 통신 과정에서 발생하는 사소한 설정 미흡에서 기인합니다.

## 2. 사용자 확인 사항 (User Review Required)
> [!important] 
> 현재 겪고 계신 **구체적인 에러 메시지나 상황**을 알려주시면 더욱 정밀한 가이드가 가능합니다. (예: "Tool not found", "Failed to start server", "JSON Parse error" 등)

## 3. 제안하는 트러블슈팅 단계

### 3.1. 1단계: 설정 무결성 검증 (Configuration Audit)
- 윈도우 경로 이스케이프(`\\` 또는 `/`) 확인
- JSON 문법(Syntax) 검사 (누락된 콤마, 중괄호 등)
- 절대 경로(Absolute Path) 사용 여부 점검

### 3.2. 2단계: 환경 및 종속성 확인 (Environment Check)
- 가상 환경(Conda/venv) 경로의 정확한 지정
- MCP 서버 작동에 필요한 라이브러리 설치 여부
- `python` 실행 경로가 시스템 환경 변수(PATH)에 등록되었는지 확인

### 3.3. 3단계: 통신 프로토콜 점검 (Transport Layer)
- `stdout`을 활용한 로그 출력 제거 (반드시 `stderr` 사용)
- 서버 실행 시 초기 세션 연결(Handshake) 지연 여부

---

## 4. 상세 수행 계획 (Proposed Changes)

### [Component] MCP 트러블슈팅 가이드 문서 제작
- 윈도우 환경에 특화된 MCP 설정 가이드를 제공합니다.
- 흔히 발생하는 오류 케이스별 체크리스트를 포함합니다.

#### [NEW] [mcp_troubleshooting_guide_20260329.md](file:///c:/Users/GOODMAN/WHToolsBox/dev_log/mcp_troubleshooting_guide_20260329.md)

---

## 5. 오픈 질문 (Open Questions)
- **에러 발생 시점**: 에이전트를 처음 켤 때 발생하나요, 아니면 특정 도구를 실행할 때 발생하나요?
- **환경**: 현재 `Claude Desktop`을 사용 중이신가요, 아니면 다른 IDE(예: Windsurf, Cursor 등)를 사용 중이신가요?

## 6. 검증 계획 (Verification Plan)
- 제안 드리는 해결 방안을 적용한 후, 에이전트 재시작을 통해 정상적으로 도구(Tool)가 활성화되는지 확인합니다.
- `npx @modelcontextprotocol/inspector`를 활용하여 서버의 독립 구동 여부를 자가 테스트합니다.

---
**WHTOOLS** 드림.
