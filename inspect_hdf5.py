import h5py

file_path = "/home/ubuntu/DASPS_Database/Preprocessed data .mat/S01preprocessed.mat"

def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"DATASET: {name} | shape={obj.shape} | dtype={obj.dtype}")
    else:
        print(f"GROUP:   {name}")

with h5py.File(file_path, "r") as f:
    print("Root keys:")
    print(list(f.keys()))

    print("\nFull structure:")
    f.visititems(print_structure)
