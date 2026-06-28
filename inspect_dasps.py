import os
import h5py

path = "/home/ubuntu/DASPS_Database"

print("Folder:", path)
print("=" * 50)

# Folder structure
for root, dirs, files in os.walk(path):
    level = root.replace(path, "").count(os.sep)
    indent = "  " * level

    print(f"{indent}{os.path.basename(root)}/")

    for f in sorted(files)[:20]:
        full = os.path.join(root, f)
        print(f"{indent}  {f} ({os.path.getsize(full)//1024} KB)")

    if len(files) > 20:
        print(f"{indent}  ... আরও {len(files)-20} টা ফাইল")

print("=" * 50)

# Find all MAT files
mat_files = []
for root, dirs, files in os.walk(path):
    for f in files:
        if f.endswith(".mat"):
            mat_files.append(os.path.join(root, f))

print("মোট .mat ফাইল:", len(mat_files))

# Inspect first MAT file using h5py
if mat_files:
    first_file = mat_files[0]

    print("\nপ্রথম MAT ফাইল:")
    print(first_file)

    try:
        with h5py.File(first_file, "r") as f:
            print("\nRoot keys:")
            print(list(f.keys()))

            print("\nContents:")

            def show(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(
                        f"DATASET: {name} | shape={obj.shape} | dtype={obj.dtype}"
                    )
                else:
                    print(f"GROUP:   {name}")

            f.visititems(show)

    except Exception as e:
        print("ফাইল পড়তে সমস্যা:", e)
