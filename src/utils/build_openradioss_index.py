"""
OpenRadioss 소스코드, 입력 파일 예제(.rad) 및 지식 문서(.md) 고속 RAG 인덱서 빌더 스크립트
Assisted by WHTOOLs OpenRadiossent
"""
import os
import re
import sqlite3
import numpy as np

# RAG용 임베딩 키워드 정의 (OpenRadioss의 핵심 물리/해석 키워드 포함)
KEYWORDS = [
    "starter", "engine", "law2", "law27", "contact", "interface", "timestep", 
    "implicit", "explicit", "mass_scaling", "subroutine", "function", "bc", 
    "boundary", "node", "element", "material", "acceleration", "velocity", 
    "stress", "strain", "energy", "damage", "rupture", "failure"
]

class LightweightCodeRAGBuilder:
    def __init__(self, db_path: str = "openradioss_rag.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_sqlite_db()
        
    def _init_sqlite_db(self):
        """SQLite FTS5 및 임베딩 메타 테이블 초기화"""
        cursor = self.conn.cursor()
        # FTS5 가상 텍스트 테이블 생성 (코드 및 문서 FTS 고속 검색용)
        cursor.execute("DROP TABLE IF EXISTS code_fts;")
        cursor.execute("CREATE VIRTUAL TABLE code_fts USING fts5(filepath, component, content);")
        # 벡터 코사인 유사도 연산을 위한 임베딩 및 메타데이터 테이블 생성
        cursor.execute("DROP TABLE IF EXISTS code_embeddings;")
        cursor.execute("""
            CREATE TABLE code_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath TEXT,
                component TEXT,
                content TEXT,
                embedding_blob BLOB
            );
        """)
        self.conn.commit()
        
    def generate_naive_embedding(self, text: str) -> np.ndarray:
        """텍스트에서 나이브 TF-IDF 기하 벡터 매핑 생성"""
        vector = np.zeros(len(KEYWORDS))
        low_text = text.lower()
        for idx, key in enumerate(KEYWORDS):
            vector[idx] = low_text.count(key)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def add_to_database(self, filepath: str, component: str, content: str):
        """데이터베이스에 FTS 텍스트 및 벡터 임베딩 저장"""
        cursor = self.conn.cursor()
        # FTS 인덱스 주입
        cursor.execute(
            "INSERT INTO code_fts (filepath, component, content) VALUES (?, ?, ?);",
            (filepath, component, content)
        )
        # 벡터 디비 주입
        emb = self.generate_naive_embedding(content)
        emb_blob = emb.tobytes()
        cursor.execute(
            "INSERT INTO code_embeddings (filepath, component, content, embedding_blob) VALUES (?, ?, ?, ?);",
            (filepath, component, content, emb_blob)
        )

    def index_source_file(self, absolute_path: str, relative_path: str):
        """파일 확장자에 맞춘 구문 분석 및 청킹 인덱싱"""
        try:
            with open(absolute_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                # 윈도우 한글 파일 백업 폴백 (CP949)
                with open(absolute_path, "r", encoding="cp949") as f:
                    content = f.read()
            except Exception as e:
                print(f"파일 디코딩 실패 건너뜀 ({relative_path}): {e}")
                return

        # 마크다운 (.md) 지식 문서 파싱 (헤더 기반 청킹)
        if relative_path.endswith(".md"):
            # #, ##, ###, #### 등으로 시작하는 라인 기준으로 분할
            sections = re.split(r"(?=^#{1,4}\s+)", content, flags=re.MULTILINE)
            section_idx = 0
            for sec in sections:
                cleaned = sec.strip()
                if not cleaned:
                    continue
                # 첫 줄에서 섹션 명칭 추출
                first_line = cleaned.split("\n")[0]
                sec_name = first_line.replace("#", "").strip()
                comp_name = f"MarkdownSection:{sec_name}" if sec_name else f"MarkdownBlock:{section_idx}"
                self.add_to_database(relative_path, comp_name, cleaned)
                section_idx += 1

        # Fortran 소스 파일 파싱 (Subroutine/Function 단위 견고한 라인 스캔 파서)
        elif relative_path.endswith((".f", ".for", ".f90")):
            lines = content.splitlines()
            in_block = False
            current_block = []
            block_name = "ModuleSource"
            block_type = "Subroutine"
            
            # 정규식 컴파일
            sub_start_pat = re.compile(r"^\s*(subroutine|function)\s+(\w+)", re.IGNORECASE)
            sub_end_pat = re.compile(r"^\s*end\s*(subroutine|function|(\s+\w+))?", re.IGNORECASE)
            
            for line in lines:
                stripped = line.strip()
                # Fortran 주석 무시 (F77: C, *, F90: !)
                is_comment = False
                if stripped:
                    if stripped.startswith("!"):
                        is_comment = True
                    elif relative_path.endswith((".f", ".for")) and line[0].lower() in ('c', '*'):
                        is_comment = True
                
                # 블록 시작 체크
                if not in_block and not is_comment:
                    start_match = sub_start_pat.match(stripped)
                    if start_match:
                        in_block = True
                        block_type = start_match.group(1).capitalize()
                        block_name = start_match.group(2)
                        current_block = [line]
                        continue
                
                if in_block:
                    current_block.append(line)
                    # 블록 종료 체크 (end subroutine, end function, end [subroutine_name], end)
                    if not is_comment and sub_end_pat.match(stripped):
                        # 주입
                        self.add_to_database(relative_path, f"{block_type}:{block_name}", "\n".join(current_block))
                        in_block = False
                        current_block = []
                else:
                    if stripped and not is_comment:
                        # 모듈 수준의 선언이나 기타 코드는 개별 한 줄씩이 아니라, 누적했다가 파일 끝날 때 처리하도록 백업 버퍼링하거나
                        # 일단 스킵하지 않고 백업 블록에 모읍니다.
                        current_block.append(line)
            
            # 남은 블록 처리
            if current_block:
                comp_lbl = f"{block_type}:{block_name}" if in_block else "ModuleSource"
                self.add_to_database(relative_path, comp_lbl, "\n".join(current_block))
                
        # C/C++ 및 Python 소스코드 파싱
        elif relative_path.endswith((".c", ".h", ".cpp", ".py")):
            # 클래스 또는 함수 단위 가벼운 구문 파싱 (공백 단락 기준 분할)
            chunks = content.split("\n\n")
            chunk_idx = 0
            for chunk in chunks:
                cleaned = chunk.strip()
                if not cleaned:
                    continue
                # 유의미한 코드 덩어리만 인덱싱 (2줄 이상 또는 기호 포함)
                if len(cleaned.split("\n")) > 1 or any(k in cleaned.lower() for k in KEYWORDS):
                    self.add_to_database(relative_path, f"CodeBlock:{chunk_idx}", cleaned)
                    chunk_idx += 1
                    
        # OpenRadioss 입력 카드 파일 (.rad) 파싱
        elif relative_path.endswith(".rad"):
            # 입력 카드는 # 또는 /로 시작하는 마커를 기준으로 분할 (특히 # 기호가 카드의 대분류)
            cards = re.split(r"(?=^#)", content, flags=re.MULTILINE)
            card_idx = 0
            for card in cards:
                cleaned = card.strip()
                if cleaned.startswith("#"):
                    first_line = cleaned.split("\n")[0]
                    # 카드 헤더명으로 컴포넌트 이름 정의 (예: RadCard:#/MAT/PLAS_JOHNS)
                    self.add_to_database(relative_path, f"RadCard:{first_line}", cleaned)
                    card_idx += 1

    def finalize(self):
        self.conn.commit()
        self.conn.close()

def build_index(doc_dir: str, db_path: str):
    print("==================================================")
    print(" WHTOOLs OpenRadiossent RAG 인덱스 DB 빌드를 시작합니다.")
    print(f" 소스 디렉토리: {doc_dir}")
    print(f" 데이터베이스 경로: {db_path}")
    print("==================================================")
    
    builder = LightweightCodeRAGBuilder(db_path)
    total_indexed = 0
    
    # doc 폴더 하위를 순회하며 기호와 파일 추출 (마크다운 파일 .md 포함)
    for root, dirs, files in os.walk(doc_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in [".f", ".for", ".f90", ".c", ".h", ".cpp", ".py", ".rad", ".md"]:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, doc_dir)
                builder.index_source_file(abs_path, rel_path)
                total_indexed += 1
                if total_indexed % 200 == 0:
                    print(f" 진행 상태: 현재 {total_indexed}개 파일 색인 완료...")
                    
    builder.finalize()
    print("==================================================")
    print(" WHTOOLs OpenRadiossent RAG 인덱스 DB 구축 완료!")
    print(f" 최종 인덱싱된 파일 수: {total_indexed} 개")
    print(f" 데이터베이스 파일 'openradioss_rag.db' 가 성공적으로 생성되었습니다.")
    print("==================================================")

if __name__ == "__main__":
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(current_dir, "../.."))
    
    # 1. 환경 변수 체크 -> 2. CLI 아규먼트 체크 -> 3. 기본 상대경로 폴백
    doc_folder = os.environ.get("WHT_OPENRADIOSS_DOC_DIR")
    if not doc_folder:
        if len(sys.argv) > 1:
            doc_folder = sys.argv[1]
        else:
            doc_folder = os.path.join(workspace_root, "wht_openradiossent_doc")
            
    db_out_path = os.environ.get("WHT_OPENRADIOSS_RAG_DB")
    if not db_out_path:
        if len(sys.argv) > 2:
            db_out_path = sys.argv[2]
        else:
            db_out_path = os.path.join(workspace_root, "openradioss_rag.db")
            
    # 절대 경로화
    doc_folder = os.path.abspath(doc_folder)
    db_out_path = os.path.abspath(db_out_path)
    
    build_index(doc_folder, db_out_path)
