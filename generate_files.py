import os

def rename_files():

    # Set the directory containing your .mov files
    directory = '/home/skylar/Documents/SkateboardML/Tricks/Varial/'  # Change to your target directory

    # Set the new base name for the files
    base_name = 'Varial'

    # Get all .mov files in the directory
    mov_files = [f for f in os.listdir(directory) if f.lower().endswith('.mov')]

    # Sort for consistency
    mov_files.sort()

    # Rename files
    for idx, filename in enumerate(mov_files, start=1):
        new_name = f"{base_name}_{idx:03}.mov"
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed '{filename}' to '{new_name}'")

    print("Renaming complete.")

def make_train_test_list():

    base_directory = '/home/skylar/Documents/SkateboardML/Tricks/'  # Change to your target directory

    trick_folders = ["Back180", "Front180", "Frontshuvit", "Kickflip", "Ollie", "Shuvit", "Varial"]

    train_files = []
    test_files = []

    for folder in trick_folders:
        full_path = os.path.join(base_directory, folder)
        print(full_path)
        
        if not os.path.isdir(full_path):
            continue

        files = sorted(os.listdir(full_path))
        files = [f for f in files if os.path.isfile(os.path.join(full_path, f))]

        midpoint = len(files) // 2

        test_files.extend([f"{folder}/{name}" for name in files[:midpoint]])
        train_files.extend([f"{folder}/{name}" for name in files[midpoint:]])

    # Write to files
    with open("testlist03.txt", "w") as test_file:
        for item in test_files:
            test_file.write(item + "\n")

    with open("trainlist03.txt", "w") as train_file:
        for item in train_files:
            train_file.write(item + "\n")

make_train_test_list()