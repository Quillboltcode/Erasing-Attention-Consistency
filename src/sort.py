data = []
with open('../log/05-02-24.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('Loss:'):
            loss, path = line.split('Path: ')
            loss = float(loss.split(': ')[1])
            path = path.strip()
            data.append((loss, path))

sorted_data = sorted(data, key=lambda x: x[0], reverse=True)

for loss, path in sorted_data:
    print(f"Loss: {loss:.4f}\tPath: {path}")