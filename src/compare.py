with open('../log/mediapipe-train.txt', 'r') as f1, open('../log/18-01-24.txt', 'r') as f2:
    file1_text = f1.read()
    file2_text = f2.read()

file1_path = file1_text.strip().split(': ')[1]
file2_paths = [line.split('\tPath: ')[1].strip() for line in file2_text.split('\n') if 'Path:' in line]

file1_paths = [line.split(': ')[1].strip() for line in file1_text.split('\n') if 'image:' in line]
file2_paths = [line.split('\tPath: ')[1].strip() for line in file2_text.split('\n') if 'Path:' in line]

print(f"Number of paths in mediapipe-train.txt: {len(file1_paths)}")
print(f"Number of paths in 05-02-24.txt: {len(file2_paths)}")

outline = []
for file1_path in file1_paths:
    if file1_path in file2_paths:
        # print(f"Path '{file1_path}' exists in file2.txt")
        continue
    else:
        # print(f"Path '{file1_path}' does not exist in file2.txt")
        outline.append(file1_path)

with open('../log/compare.txt', 'w') as f:
    for path in outline:
        f.write(f"Path: {path}\n")

print(f"Number of paths not in mediapipe-train.txt: {len(file2_paths)}")
print(f"Number of paths not in 05-02-24.txt: {len(outline)}")