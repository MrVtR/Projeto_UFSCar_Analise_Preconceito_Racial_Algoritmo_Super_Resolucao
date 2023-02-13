#importing cv2 and matplotlid
import cv2
import matplotlib.pyplot as plt
import os
from pprint import pprint
import shutil
from deepface import DeepFace
import timeit

filesPath = []
def lista_arquivos_em_pasta(src_folder):
    for dirpath, dirnames, filenames in os.walk(src_folder):
        for filename in filenames:
            filesPath.append(os.path.join(dirpath, filename))

def verifica_se_pasta_existe(directory_name):
    if not os.path.isdir(directory_name):
        os.makedirs(directory_name)

def move_arquivo_para_pasta(file_path, directory_name):
    shutil.move(file_path, os.path.join(directory_name, os.path.basename(file_path)))

pasta_origem = '.\\test'
lista_arquivos_em_pasta(pasta_origem)

starttime = timeit.default_timer()
print("The start time is :",starttime)

for path in filesPath:
    try:
        img =  cv2.imread(path)

        plt.imshow(img) 
        color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        prediction = DeepFace.analyze(color_img,enforce_detection=False,silent=True)

        nomePasta = '.\\raças\\'+prediction[0]['dominant_race']
        verifica_se_pasta_existe(nomePasta)
        
        move_arquivo_para_pasta(path, nomePasta)
        print(f"O arquivo '{path}' foi movido para '{nomePasta}'")
    except(ValueError):
        print("Arquivo:",path,"- Rosto não detectado")
        pass
print("The time difference is :", timeit.default_timer() - starttime)
