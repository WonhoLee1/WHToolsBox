# -*- mode: python ; coding: utf-8 -*-
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# 주요 라이브러리들의 데이터 파일(바이너리, 설정 파일 등)을 수집합니다.
mujoco_datas = collect_data_files('mujoco')
jax_datas = collect_data_files('jaxlib')
scipy_datas = collect_data_files('scipy')
pyvista_datas = collect_data_files('pyvista')

# 누락되기 쉬운 동적 임포트 패키지들을 모두 포함합니다.
hidden_imports = []
hidden_imports += collect_submodules('mujoco')
hidden_imports += collect_submodules('PySide6')
hidden_imports += collect_submodules('rich')
hidden_imports += collect_submodules('pyqtgraph')
hidden_imports += collect_submodules('scipy')
hidden_imports += collect_submodules('pyvista')
hidden_imports += collect_submodules('jax')
hidden_imports += collect_submodules('jaxlib')
hidden_imports += ['xml.etree.ElementTree', 'concurrent.futures', 'openpyxl', 'pandas']

a = Analysis(
    ['run_drop_simulation_cases_v6.py'],
    pathex=['.'], # 현재 경로를 탐색 경로에 추가 (하위 폴더 모듈 인식용)
    binaries=[],
    datas=[] + mujoco_datas + jax_datas + scipy_datas + pyvista_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter', 'unittest', 'pytest', 'IPython', 'notebook', 'jedi',
        'matplotlib.tests', 'scipy.tests', 'numpy.random._examples',
        'pip', 'setuptools', 'wheel', 'cython', 'PyQt5', 'PySide2',
        'black', 'mypy', 'sphinx', 'numpydoc', 'nbconvert', 'nbformat'
    ], # 불필요한 개발/테스트/GUI 툴킷 모듈을 제외하여 용량을 최소화합니다.
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='WHTools_DropSimulator_v6',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True, # 오류 확인 및 터미널 출력(rich 등)을 위해 True로 유지하는 것을 권장합니다.
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='app_icon.ico' # 아이콘 파일이 있다면 주석 해제 후 적용하세요
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='WHTools_DropSimulator_v6',
)
