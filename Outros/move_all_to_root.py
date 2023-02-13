import os
import shutil

def move_files_to_main_folder(src_folder):
    for dirpath, dirnames, filenames in os.walk(src_folder):
        for filename in filenames:
            src_file = os.path.join(dirpath, filename)
            dst_file = os.path.join(src_folder, filename)
            shutil.move(src_file, dst_file)
        
        if dirpath != src_folder:
            print(dirpath.split('\\')[1],"movido")
            shutil.rmtree(dirpath)


src_folder = './test'
move_files_to_main_folder(src_folder)
print("Sucesso")