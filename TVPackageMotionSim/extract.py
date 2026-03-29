
import sys

def extract_lines(file_path, start, count):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for i in range(start-1, min(start-1+count, len(lines))):
                print(f"{i+1}: {lines[i]}", end='')
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    extract_lines(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
