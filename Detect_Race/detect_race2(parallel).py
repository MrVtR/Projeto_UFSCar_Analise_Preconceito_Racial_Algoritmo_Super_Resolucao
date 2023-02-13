import cv2
import matplotlib.pyplot as plt
import os
import shutil
import multiprocessing
from deepface import DeepFace
from multiprocessing import Process
import timeit

def list_files_in_folder(src_folder):
    files_path = []
    for dirpath, dirnames, filenames in os.walk(src_folder):
        for filename in filenames:
            files_path.append(os.path.join(dirpath, filename))
    return files_path

def ensure_directory_exists(directory_name):
    if not os.path.isdir(directory_name):
        os.makedirs(directory_name)

def move_file_to_directory(file_path, directory_name):
    shutil.move(file_path, os.path.join(directory_name, os.path.basename(file_path)))

def analyze_and_move_file(path):
    try:
        img = cv2.imread(path)
        plt.imshow(img)
        color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        prediction = DeepFace.analyze(color_img, enforce_detection=False, silent=False)
        race_folder = '.\\raça_teste\\' + prediction[0]['dominant_race']
        ensure_directory_exists(race_folder)
        move_file_to_directory(path, race_folder)
        print(f"The file '{path}' has been moved to '{race_folder}'.")
    except(ValueError):
        print(f"File: {path} - Face not detected")
        pass

if __name__ == '__main__':
    src_folder = '.\\teste'
    files_path = list_files_in_folder(src_folder)

    starttime = timeit.default_timer()
    print("The start time is :",starttime)
    # for path in files_path:
    #     p = Process(target=analyze_and_move_file, args=(path,))
    #     p.start()
    #     p.join()
    with multiprocessing.Pool(processes=os.cpu_count()) as pool: 
        pool.map(analyze_and_move_file, [path for path in files_path])
    print("The time difference is :", timeit.default_timer() - starttime)

# with multiprocessing.Pool(processes=os.cpu_count()) as pool:
#     pool.map(analyze_and_move_file, files_path)