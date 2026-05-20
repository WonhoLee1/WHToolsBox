"""Inspect box.msh node section format."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('test_box_msh/box.msh', 'r') as f:
    lines = f.readlines()

# Print first 20 lines of Nodes section
in_nodes = False
count = 0
print('=== $Nodes section (first 10 lines) ===')
for line in lines:
    stripped = line.strip()
    if stripped == '$Nodes':
        in_nodes = True
        continue
    if stripped == '$EndNodes':
        break
    if in_nodes:
        print(repr(stripped))
        count += 1
        if count >= 10:
            break

# Print file header
print('\n=== File header ===')
for line in lines[:5]:
    print(repr(line.strip()))
