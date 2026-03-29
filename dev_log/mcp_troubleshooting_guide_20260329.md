# MCP (Model Context Protocol) 윈도우 상세 해결 가이드

안녕하세요, **WHTOOLS**입니다. 

승인 주신 계획에 따라, MCP 설정 및 운영 중 발생하는 고질적인 오류들을 해결하기 위한 **심화 가이드**를 작성하였습니다. 윈도우(Windows) 기반 공학 시뮬레이션 환경(MuJoCo 등)을 다루시는 만큼, 환경 변수와 경로 관리에 초점을 맞추어 정리했습니다.

---

## 1. 윈도우 경로 및 JSON 이스케이프 (Path & Syntax)

윈도우 환경에서 `claude_desktop_config.json`을 수정할 때 가장 많이 범하는 실수는 경로 구분자(`\`)의 미처리입니다. 

### 1.1. 안전한 경로 지정 방법
JSON 문자열 내부에서 단일 역슬래시(`\`)는 이스케이프 문자로 인식됩니다. 따라서 다음 중 하나의 방법을 선택하십시오.

- **방법 1: 슬래시(/) 사용 (권장)**: 윈도우에서도 대부분의 현대적인 프로그래밍 환경은 슬래시를 자동 인식합니다.
  ```json
  "command": "C:/Users/GOODMAN/.conda/envs/mcp/python.exe"
  ```
- **방법 2: 이중 역슬래시(\\\\) 사용**:
  ```json
  "command": "C:\\Users\\GOODMAN\\.conda\\envs\\mcp\\python.exe"
  ```

### 1.2. 공백 포함 경로 처리
경로에 `Program Files`와 같이 공백이 있다면, 전체 경로를 따옴표로 감싸는 것에 주의해야 합니다. 
`args` 리스트에 인자를 넣을 때는 공백이 있어도 하나의 인덱스로 취급되므로 별도의 따옴표 이중 처리는 필요 없으나, `command` 자체에 공백이 있다면 가능한 공백이 없는 경로를 사용하거나 심볼릭 링크를 권장합니다.

---

## 2. 가상 환경(Conda/venv) 연동의 핵심

터미널에서는 `conda activate`로 해결되지만, 클로드(Claude)는 해당 쉘 환경을 알지 못합니다. 따라서 **가상 환경 내의 `python.exe` 절대 경로**를 직접 지정해야 합니다.

### 2.1. Conda 환경 경로 확인
```powershell
# 가상 환경 경로 확인용 명령어
conda env list
```
확인된 경로의 `\python.exe`를 `command` 항목으로 사용하십시오.

### 2.2. 예시 설정 (`claude_desktop_config.json`)
```json
{
  "mcpServers": {
    "WHToolsServer": {
      "command": "C:/Users/GOODMAN/.conda/envs/simulation/python.exe",
      "args": [
        "C:/Users/GOODMAN/WHToolsBox/mcp_server.py"
      ],
      "env": {
        "PYTHONPATH": "C:/Users/GOODMAN/WHToolsBox"
      }
    }
  }
}
```

---

## 3. "Standard I/O Error" 및 JSON Parse Error 방지 (`stderr` 사용)

MCP는 **표준 출력(stdout)**을 데이터 통로로 사용합니다. 따라서 서버 코드(`python` 등) 내부에서 발생하는 모든 디버그 출력은 **표준 에러(stderr)**로 내보내야 합니다.

### 3.1. 올바른 로그 출력 기법 (Python)
> [!caution] 
> `print()`를 무심코 사용하면 클라이언트와의 연결이 끊어지며 "MCP Error"가 발생합니다.

```python
import sys

# 오류 유발 코드 (stdout 오염)
# print("Server started...") 

# 올바른 코드 (stderr 사용)
print("Server started successfully", file=sys.stderr)
```

---

## 4. MCP Inspector를 활용한 자가 진단

설정이 완벽해 보이는데도 에러가 난다면, 클라이언트를 탓하기 전 **Inspector**로 서버 자체의 무결성을 검증하십시오.

1. **설치/실행**:
   ```powershell
   npx @modelcontextprotocol/inspector C:/Users/GOODMAN/.conda/envs/mcp/python.exe C:/Users/GOODMAN/WHToolsBox/mcp_server.py
   ```
2. **확인 사항**:
   - `Resources`, `Prompts`, `Tools` 탭이 정상적으로 활성화되는가?
   - 도구를 호출했을 때 `stderr`에 찍히는 로그가 없는가?

---

## 5. 마치며 (Summary)

> [!tip] 
> **3줄 요약**
> 1. 경로는 `/`를 사용하거나 `\\`를 사용하여 이스케이프하십시오.
> 2. `python.exe`는 반드시 가상 환경 내부의 **절대 경로**를 사용하십시오.
> 3. 코드 내의 모든 출력은 `file=sys.stderr`를 통해 내보내십시오.

설정을 변경하신 후에는 반드시 **Claude Desktop을 완전히 종료(System Tray에서도 종료)**한 후 다시 실행해 주시기 바랍니다. 

문제가 지속될 경우, `%APPDATA%\Claude\logs\mcp.log` 파일의 내용을 공유해 주시면 즉각 분석해 드리겠습니다.

진행 중인 **MuJoCo 시뮬레이션 자동화** 업무에 차질이 없으시길 바랍니다.

**WHTOOLS** 드림.
