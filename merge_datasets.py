import os
import shutil

source_dirs = ['data\\data', 'Foto bosmina\\Foto bosmina']
target_dir = 'data_full'

os.makedirs(target_dir, exist_ok=True)

for src in source_dirs:
    for class_name in os.listdir(src):
        class_src = os.path.join(src, class_name)
        class_dst = os.path.join(target_dir, class_name)

        if os.path.isdir(class_src):
            os.makedirs(class_dst, exist_ok=True)
            for fname in os.listdir(class_src):
                src_path = os.path.join(class_src, fname)
                dst_path = os.path.join(class_dst, fname)

                if os.path.isfile(src_path):
                    if not os.path.exists(dst_path):
                        shutil.copy2(src_path, dst_path)