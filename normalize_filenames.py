import os
import re
import argparse

def normalize_name(filename, is_dir=False):
    if is_dir:
        name = filename
        ext = ""
    else:
        name, ext = os.path.splitext(filename)
    
    # 1. Lowercase
    name = name.lower()
    
    # 2. Handle chemical formula markup (sub/sup)
    name = name.replace("_sub_", "_").replace("_sup_", "_")
    
    # 3. Replace all whitespace (spaces, tabs, newlines) with underscores
    name = re.sub(r'\s+', '_', name)
    
    # 4. Remove/Replace special characters
    # Keep alphanumeric, underscores, and dashes. Replace everything else (including dots) with underscores.
    name = re.sub(r'[^a-zA-Z0-9\-_]', '_', name)
    
    # 5. Deduplicate underscores
    name = re.sub(r'_+', '_', name)
    
    # 6. Strip leading/trailing underscores
    name = name.strip('_')
    
    if is_dir:
        return name
    else:
        # Standardize extension to lowercase and remove non-alphanumeric if any
        clean_ext = ext.lower().strip()
        return name + clean_ext

def process_directory(directory, execute=False):
    renames = []
    
    # Walk top-down=False to rename children before parents
    for root, dirs, files in os.walk(directory, topdown=False):
        # Process files
        for f in files:
            if f.startswith('.'):
                continue
            new_name = normalize_name(f, is_dir=False)
            if new_name != f:
                old_path = os.path.join(root, f)
                new_path = os.path.join(root, new_name)
                renames.append((old_path, new_path))
        
        # Process directories
        for d in dirs:
            if d.startswith('.'):
                continue
            new_name = normalize_name(d, is_dir=True)
            if new_name != d:
                old_path = os.path.join(root, d)
                new_path = os.path.join(root, new_name)
                renames.append((old_path, new_path))
                
    if not renames:
        print("No items need further normalization.")
        return

    print(f"Proposed {len(renames)} renames:")
    for old, new in renames:
        print(f"  '{old}'\n    -> '{new}'")

    if execute:
        print("\nExecuting renames...")
        for old_path, new_path in renames:
            if os.path.exists(new_path) and old_path.lower() != new_path.lower():
                print(f"Warning: Destination {new_path} already exists. Skipping.")
                continue
            
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {os.path.basename(old_path)} -> {os.path.basename(new_path)}")
            except Exception as e:
                print(f"Error renaming {old_path}: {e}")
        print("Done.")
    else:
        print("\nDry run completed. Use --execute to apply changes.")

def main():
    parser = argparse.ArgumentParser(description="Normalize filenames and folder names in a directory.")
    parser.add_argument("directory", help="Path to the directory to process.")
    parser.add_argument("--execute", action="store_true", help="Actually perform the renaming.")
    args = parser.parse_args()

    if not os.path.exists(args.directory):
        print(f"Error: {args.directory} does not exist.")
        return

    process_directory(args.directory, args.execute)

if __name__ == "__main__":
    main()
