"""
CalculiX 소스코드 및 입력 파일 예제 고속 RAG 인덱서 빌더 스크립트
Assisted by WHTOOLs Calculixent
"""
import os
import re
import sqlite3
import numpy as np

# RAG용 임베딩 키워드 정의 (CalculiX의 핵심 물리/해석 키워드 포함)
KEYWORDS = [
    "subroutine", "function", "arpack", "spooles", "pastix", "pardiso", "stiffness", 
    "mass", "eigenvalue", "boundary", "shell", "solid", "contact", "static", "frequency", 
    "dynamic", "heat", "thermal", "fluid", "electromagnetics", "equation", "umat", "elset", "nset"
]

class LightweightCodeRAGBuilder:
    def __init__(self, db_path: str = "calculix_rag.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_sqlite_db()
        
    def _init_sqlite_db(self):
        """SQLite FTS5 및 임베딩 메타 테이블 초기화"""
        cursor = self.conn.cursor()
        # FTS5 가상 텍스트 테이블 생성 (코드 FTS 고속 검색용)
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
        """텍스트에서 나이브 TF-IDF 기하 백터 매핑 생성"""
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

        # Fortran 소스 파일 파싱 (Subroutine 단위 분할)
        if relative_path.endswith((".f", ".for", ".f90")):
            # Fortran 서브루틴 및 펑션 추출
            subroutines = re.findall(
                r"((?:subroutine|function)\s+\w+.*?end\s+(?:subroutine|function))", 
                content, 
                re.IGNORECASE | re.DOTALL
            )
            if subroutines:
                for sub in subroutines:
                    first_line = sub.split("\n")[0]
                    name_match = re.search(r"(?:subroutine|function)\s+(\w+)", first_line, re.IGNORECASE)
                    comp_name = f"Subroutine:{name_match.group(1)}" if name_match else "Subroutine"
                    self.add_to_database(relative_path, comp_name, sub.strip())
            else:
                # 서브루틴이 없는 경우 파일 전체를 하나의 단위로 주입
                self.add_to_database(relative_path, "ModuleSource", content.strip())
                
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
                    
        # Abaqus/CalculiX 입력 카드 파일 (.inp) 파싱
        elif relative_path.endswith(".inp"):
            # 입력 카드는 * 마커를 기준으로 분할
            cards = re.split(r"(?=\*)", content)
            card_idx = 0
            for card in cards:
                cleaned = card.strip()
                if cleaned.startswith("*"):
                    first_line = cleaned.split("\n")[0]
                    self.add_to_database(relative_path, f"InpCard:{first_line}", cleaned)
                    card_idx += 1

    def finalize(self):
        self.conn.commit()
        self.conn.close()

def build_index(doc_dir: str, db_path: str):
    print("==================================================")
    print(" WHTOOLs Calculixent RAG 인덱스 DB 빌드를 시작합니다.")
    print(f" 소스 디렉토리: {doc_dir}")
    print(f" 데이터베이스 경로: {db_path}")
    print("==================================================")
    
    builder = LightweightCodeRAGBuilder(db_path)
    total_indexed = 0
    
    # doc 폴더 하위를 순회하며 기호와 파일 추출
    for root, dirs, files in os.walk(doc_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in [".f", ".for", ".f90", ".c", ".h", ".cpp", ".py", ".inp"]:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, doc_dir)
                builder.index_source_file(abs_path, rel_path)
                total_indexed += 1
                if total_indexed % 100 == 0:
                    print(f" 진행 상태: 현재 {total_indexed}개 소스코드 파일 색인 완료...")
                    
    builder.finalize()
    print("==================================================")
    print(" WHTOOLs Calculixent RAG 인덱스 DB 구축 완료!")
    print(f" 최종 인덱싱된 소스 파일 수: {total_indexed} 개")
    print(f" 데이터베이스 파일 'calculix_rag.db' 가 성공적으로 생성되었습니다.")
    print("==================================================")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(current_dir, "../.."))
    doc_folder = os.path.join(workspace_root, "wht_calculixent_doc")
    db_out_path = os.path.join(workspace_root, "calculix_rag.db")
    
    build_index(doc_folder, db_out_path)

