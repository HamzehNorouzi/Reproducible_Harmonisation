import os
def get_actual_path(base_path, marker):
    """Dynamically searches the base_path to find the exact folder containing the marker."""
    for root, dirs, files in os.walk(base_path):
        if marker in dirs or marker in files:
            return root
    print(f"WARNING: Could not find '{marker}' inside {base_path}!")
    return base_path
