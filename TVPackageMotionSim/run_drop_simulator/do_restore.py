import json
import re

with open('recovered.txt', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    if '"step_index":282' in line:
        # JSON parsing failed due to control characters (like newlines inside strings in raw log lines).
        # We can extract the ReplacementContent using regular expression as a robust fallback.
        match = re.search(r'"ReplacementContent"\s*:\s*"(.*?)"\s*,\s*"StartLine"', line, re.DOTALL)
        if match:
            content = match.group(1)
            # Unescape JSON string escapes
            # Replace escaped quotes, newlines, tabs
            content = content.replace(r'\"', '"').replace(r'\n', '\n').replace(r'\t', '\t')
            with open('restore_output.py', 'w', encoding='utf-8') as out:
                out.write(content)
            print("Success via Regex")
            break
        else:
            print("Regex pattern not matched")
