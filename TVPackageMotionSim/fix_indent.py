import sys

with open("c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_v3.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    if line.strip().startswith("def _main_loop(self, total_steps):"):
        start_idx = idx
        break

# Find the end of _main_loop
end_idx = start_idx
for idx in range(start_idx + 1, len(lines)):
    if lines[idx].startswith("    def "):
        end_idx = idx
        break

# We want to indent everything inside _main_loop by 4 spaces, BUT we also need to inject `while True:`
# Original code:
#     def _main_loop(self, total_steps):
#         """핵심 시뮬레이션 물리 연산 루프 (Step/Jump/Update)"""
#         # 헤더 출력
#         self._print_terminal_header()
        
new_lines = lines[:start_idx+1]
new_lines.append(lines[start_idx+1]) # docstring

new_lines.append("        while True:\n")
for i in range(start_idx+2, end_idx):
    if lines[i].strip() == "":
        new_lines.append("\n")
    else:
        new_lines.append("    " + lines[i])

new_lines.extend(lines[end_idx:])

# Now we find the place where it breaks out of the loop at the very end (instead of continue)
# and we change `continue` to nothing (since it's in a while loop now, `continue` works, but wait.
# Actually, if it's in `while True:`, `continue` WILL WORK!
# But wait, original code:
#             if self.ctrl_reset_request:
#                 self.ctrl_reset_request = False
#                 continue
# Because we added `while True:`, this `continue` is now structurally inside `while True:`.
# We also need to add `break` if it is NOT reset_request.
# Let's fix the specific logic at the end of the `if current_step_cnt >= total_steps:` block.

# Let's just write back the lines first, then we can do another fix.
with open("c:/Users/GOODMAN/WHToolsBox/TVPackageMotionSim/run_drop_simulation_v3.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)
