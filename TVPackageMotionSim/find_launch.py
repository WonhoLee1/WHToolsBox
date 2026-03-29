import os

filepath = r'c:\Users\GOODMAN\WHToolsBox\TVPackageMotionSim\run_drop_simulation_v3.py'
with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
    for i, line in enumerate(f):
        if 'launch_passive' in line:
            print(f"Line {i+1}: {line.strip()}")
