import os

root = r"C:\project\archive\hagrid-classification-512p"
total = 0

for c in os.listdir(root):
    p = os.path.join(root, c)
    if os.path.isdir(p):
        total += len([
            f for f in os.listdir(p)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])

print("Total images =", total)
