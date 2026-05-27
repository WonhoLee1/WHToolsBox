import os
import glob
import re

def extract_date(filename):
    match = re.search(r'(\d{4}[-]?\d{2}[-]?\d{2})', filename)
    if match:
        date_str = match.group(1).replace('-', '')
        return date_str
    return "00000000"

def get_content(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        try:
            with open(filepath, 'r', encoding='cp949') as f:
                return f.read()
        except Exception as e:
            return f"Error reading {filepath}: {e}"

def compile():
    dev_log_dir = r"c:\Users\GOODMAN\WHToolsBox\TVPackageMotionSim\dev_log"
    out_file = os.path.join(dev_log_dir, "comprehensive_dev_history.md")
    
    # 1. Base Summary Content
    summary_content = """# WHToolsBox Comprehensive Development History & Technical Whitepaper

본 문서는 `TVPackageMotionSim\\dev_log` 디렉토리에 2026년 3월부터 수개월간 축적된 200여 개의 아티팩트와 마크다운 로그를 단일 문서로 완전하게 통합·정리한 핵심 백서이자 영구 보존용 아카이브입니다.

---

## I. WHToolsBox 소개 및 개요

대형 디스플레이 제품의 유통·물류 과정에서 발생하는 낙하 충격을 방지하기 위한 포장 설계는 높은 비용과 시간이 소요되는 과정입니다. **WHToolsBox**는 이를 해결하기 위해 개발된 **멀티스케일 디지털 트윈 프레임워크**입니다.

- **핵심 철학**: "빠르지만 충분히 정확한" 감차원 실시간 시뮬레이션(MuJoCo)과 "느리지만 극도로 정밀한" 상세 유한요소해석(FEA)의 유기적 결합.
- **주요 성과**: 감차원 이산-연속체 결합 모델과 JAX 기반 GPU 가속 후처리를 통해 기존 상용 FEA 대비 해석 시간을 **95% 이상 단축**하면서도 물리적 정합도를 완벽히 유지.

---

## II. 핵심 기술 문서 (Core Technical Documents)

"""
    
    # Core Files
    core_files = ["paper_20260406.md", "engineering_knowledge.md", "str_metrics_theoretical_background.md", "issue_tracker.md"]
    core_content = ""
    for cf in core_files:
        path = os.path.join(dev_log_dir, cf)
        if os.path.exists(path):
            core_content += f"\n### Core Document: {cf}\n\n"
            core_content += get_content(path)
            core_content += "\n\n---\n"
    
    # Gather all other files
    all_mds = glob.glob(os.path.join(dev_log_dir, "*.md"))
    
    impl_plans = []
    walkthroughs = []
    tasks = []
    others = []
    
    exclude_files = ["comprehensive_dev_history.md", "implementation_plan_20260528.md", "compile_history.py"] + core_files
    
    for path in all_mds:
        filename = os.path.basename(path)
        if filename in exclude_files:
            continue
            
        if "implementation_plan" in filename or "refactoring_plan" in filename or "layout_optimization_plan" in filename:
            impl_plans.append(path)
        elif "walkthrough" in filename:
            walkthroughs.append(path)
        elif "task" in filename:
            tasks.append(path)
        else:
            others.append(path)
            
    impl_plans.sort(key=lambda x: extract_date(os.path.basename(x)))
    walkthroughs.sort(key=lambda x: extract_date(os.path.basename(x)))
    tasks.sort(key=lambda x: extract_date(os.path.basename(x)))
    others.sort(key=lambda x: extract_date(os.path.basename(x)))
    
    def build_section(title, files):
        section = f"\n## {title}\n\n"
        for p in files:
            fname = os.path.basename(p)
            section += f"### Archive: {fname}\n\n"
            content = get_content(p)
            section += content + "\n\n---\n"
        return section

    archive_content = ""
    archive_content += build_section("III. Implementation Plans Archive", impl_plans)
    archive_content += build_section("IV. Walkthroughs Archive", walkthroughs)
    archive_content += build_section("V. Tasks Archive", tasks)
    archive_content += build_section("VI. Miscellaneous History Archive", others)
    
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(summary_content + core_content + archive_content)
        
    print(f"Successfully compiled {len(impl_plans)+len(walkthroughs)+len(tasks)+len(others)+len(core_files)} files into {out_file}")

if __name__ == "__main__":
    compile()
