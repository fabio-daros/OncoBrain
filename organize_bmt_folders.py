import os
import shutil

SOURCE_DIR = 'dataset/bmt/Brown Multicellular ThinPrep Database - images'
TARGET_DIR = 'dataset/bmt'

def main():
    if not os.path.exists(SOURCE_DIR):
        print("Error: Origin folder not found.")
        return

    for filename in os.listdir(SOURCE_DIR):
        if filename.endswith(".JPG") or filename.endswith(".jpg"):
            if filename.startswith("HSIL"):
                class_folder = "HSIL"
            elif filename.startswith("LSIL"):
                class_folder = "LSIL"
            elif filename.startswith("NIL"):
                class_folder = "NILM"
            else:
                continue  # ignora arquivos inesperados

            src_path = os.path.join(SOURCE_DIR, filename)
            dst_dir = os.path.join(TARGET_DIR, class_folder)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.move(src_path, os.path.join(dst_dir, filename))

    # Remover a pasta intermediária
    try:
        os.rmdir(SOURCE_DIR)
        print(f"removed folder: {SOURCE_DIR}")
    except OSError:
        print(f"Folder {SOURCE_DIR} It has not been removed (may not be empty).")

    print("Successful organized images!")

if __name__ == "__main__":
    main()
