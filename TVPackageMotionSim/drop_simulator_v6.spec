# -*- mode: python ; coding: utf-8 -*-
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# 주요 라이브러리들의 데이터 파일(바이너리, 설정 파일 등)을 수집합니다.
mujoco_datas = collect_data_files('mujoco')
jax_datas = collect_data_files('jaxlib')
scipy_datas = collect_data_files('scipy')
pyvista_datas = collect_data_files('pyvista')
glfw_datas = collect_data_files('glfw')

# 누락되기 쉬운 동적 임포트 패키지들을 모두 포함합니다.
hidden_imports = [
    'xml.etree.ElementTree', 
    'concurrent.futures', 
    'openpyxl', 
    'pandas', 
    'pkg_resources', 
    'jaraco.text',
    'lxml',
    'lxml.etree',
    'numba',
    'numba.core',
    'numba.core.entrypoints',
    'run_drop_simulator.runopenradioss', 
    'run_drop_simulator.inp2rad',
    'run_drop_simulator.whts_analysis_pipeline',
    'run_drop_simulator.whts_control_panel',
    'run_drop_simulator.whts_data',
    'run_drop_simulator.whts_engine',
    'run_drop_simulator.whts_exporter',
    'run_drop_simulator.whts_ista_helper',
    'run_drop_simulator.whts_jax_ssr',
    'run_drop_simulator.whts_mapping',
    'run_drop_simulator.whts_monitor',
    'run_drop_simulator.whts_multipostprocessor',
    'run_drop_simulator.whts_multipostprocessor_engine',
    'run_drop_simulator.whts_multipostprocessor_ui',
    'run_drop_simulator.whts_postprocess_engine_v2',
    'run_drop_simulator.whts_postprocess_ui_v2',
    'run_drop_simulator.whts_radioss_builder',
    'run_drop_simulator.whts_reporting',
    'run_drop_simulator.whts_theme',
    'run_drop_simulator.whts_utils',
    'run_drop_simulator.wht_export_sim_result',
    'run_drop_simulator.wht_plotwindowutil',
    'run_drop_simulator.wht_ui_components',
    'run_drop_simulator.mpl_extension',
    'run_discrete_builder',
    'run_discrete_builder.whtb_base',
    'run_discrete_builder.whtb_builder',
    'run_discrete_builder.whtb_config',
    'run_discrete_builder.whtb_models',
    'run_discrete_builder.whtb_physics',
    'run_discrete_builder.whtb_utils',
    'mujoco',
    'PySide6',
    'PySide6.QtCore',
    'PySide6.QtWidgets',
    'PySide6.QtGui',
    'PySide6.QtUiTools',
    'rich',
    'pyqtgraph',
    'scipy',
    'scipy.spatial',
    'scipy.spatial.transform',
    'scipy.optimize',
    'scipy.integrate',
    'scipy.interpolate',
    'pyvista',
    'pyvista.plotting',
    'pyvista.plotting.qt_plotting',
    'pyvistaqt',
    'jax',
    'jax.numpy',
    'jax.lax',
    'jaxlib',
    'glfw'
]


a = Analysis(
    ['run_drop_simulation_cases_v6.py'],
    pathex=['.'], # 현재 경로를 탐색 경로에 추가 (하위 폴더 모듈 인식용)
    binaries=[],
    datas=[
        ('sidebar_logo.png', '.'),
        ('sidebar_logo_raw.png', '.'),
        ('ui_banner.png', '.'),
        ('external_tools_config.ini', '.'),
        ('resources', 'resources'),
    ] + mujoco_datas + jax_datas + scipy_datas + pyvista_datas + glfw_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'torch', 'tensorflow', 'tensorboard', 'dask', 'distributed', 'ipython', 
        'notebook', 'jupyter', 'jupyter_rfb', 'spyder', 'pylint', 'pytest', 
        'unittest', 'jedi', 'pip', 'wheel', 'cython', 'black', 'mypy', 
        'sphinx', 'numpydoc', 'nbconvert', 'nbformat', 'matplotlib.tests', 
        'scipy.tests', 'numpy.random._examples', 'PyQt5', 'PySide2', 'tkinter'
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
    icon='resources/logo_icon.ico'
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
