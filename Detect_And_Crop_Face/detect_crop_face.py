import cv2
import os

filesPath = []
def lista_arquivos_em_pasta(src_folder):
    for dirpath, dirnames, filenames in os.walk(src_folder):
        for filename in filenames:
            filesPath.append(os.path.join(dirpath, filename))

def list_sub_directories(folder_path):
    sub_directories = []
    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            sub_directories.append(os.path.join(root, dir))
    return sub_directories

def verifica_se_pasta_existe(directory_name):
    if not os.path.isdir(directory_name):
        os.makedirs(directory_name)


folder_path = 'C:\\Users\\vitor\\Desktop\\Jurandy\\racas'
sub_directories = list_sub_directories(folder_path)

for sub_dir in sub_directories:
    lista_arquivos_em_pasta(sub_dir)

for file in filesPath:
    fileName = file.split('\\')[-1].split('.')[0]
    raca = file.split('\\')[-2]
    croppedPath = folder_path+'\\'+raca+'_cropped'
    detectedPath = folder_path+'\\'+raca+'_detected'
    verifica_se_pasta_existe(croppedPath)
    verifica_se_pasta_existe(detectedPath)

    img = cv2.imread(file)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        faces = img[y:y + h, x:x + w]
        cv2.imwrite(croppedPath+'\\'+fileName+'_face.jpg', faces)
        print(fileName,'croppado e salvo com sucesso')
        
    cv2.imwrite(detectedPath+'\\'+fileName+'_detected.jpg', img)
