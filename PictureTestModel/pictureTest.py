import cv2
import matplotlib.pyplot as plt
import utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir a webcam")

while True:
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    #Pressione a Barra de Espaço para Salvar a imagem
    if cv2.waitKey(1) & 0xFF == ord(' '):
        cv2.imwrite('foto_original.png', frame)
        modelo= None
        algoritmo = None
        fx = None
        fy = None
        file = None
        setModel = None

        while(modelo==None):
          print("Digite qual raça do modelo de melhoramento de imagem gostaria de testar:")
          print('1: Modelo treinado com fotos de Raça Asiática')
          print('2: Modelo treinado com fotos de Raça Negra')
          print('3: Modelo treinado com fotos de Raça Indiana')
          print('4: Modelo treinado com fotos de Raça Latina')
          print('5: Modelo treinado com fotos de Raça do Oriente Médio')
          print('6: Modelo treinado com fotos de Raça Branca')
          print('7: Modelo Balanceado')
          print("Digite sua escolha: ",end='')
          escolhaModelo = int(input())
          if(escolhaModelo==1):
            modelo = 'asian'
          elif(escolhaModelo==2):
            modelo = 'black'
          elif(escolhaModelo==3):
            modelo = 'indian'
          elif(escolhaModelo==4):
            modelo = 'latino'
          elif(escolhaModelo==5):
            modelo = 'middle'
          elif(escolhaModelo==6):
            modelo = 'white'
          elif(escolhaModelo==7):
            modelo = 'mixed'
          else:
            print("Escolha de Modelo inválida") 
            print()

        print()
        while(algoritmo==None):
          print("Digite qual algoritmo de melhoramento de imagem gostaria de testar:")
          print('1: TF-ESPCN')
          print('2: FSRCNN')
          print('3: TF-LapSRN')
          print("Digite sua escolha: ",end='')
          escolhaAlgoritmo = int(input())
          if(escolhaAlgoritmo ==1):
            algoritmo = 'TF-ESPCN'
            fx=4
            fy=4
            file = 'frozen_ESPCN_graph_x4.pb'
            setModel = 'espcn'
          elif(escolhaAlgoritmo==2):
            algoritmo = 'FSRCNN'
            fx=2
            fy=2
            file = 'FSRCNN_x2.pb'
            setModel = 'fsrcnn'
          elif(escolhaAlgoritmo==3):
            algoritmo='TF-LapSRN'
            fx=2
            fy=2
            file = 'LapSRN_x2.pb'
            setModel = 'lapsrn'
          else:
            print("Escolha de Algoritmo inválida") 
            print()

        print()
        print("Algoritmo Escolhido:",algoritmo)
        print("Modelo Escolhido:",modelo)
        print("FX FY:",fx,fy)

        print()
        print("Iniciando Crop da foto")

        # Recorte da face obtida
        img = cv2.imread('./foto_original.png')
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            faces = img[y:y + h, x:x + w]
            cv2.imwrite('./foto_face.jpg', faces)
            print('Foto croppada e salva com sucesso')
            
        print("Iniciando Melhoramento de Imagem")

        # Uso do SuperSampling
        img = cv2.imread("./foto_face.jpg")

        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        
        #Path, fx e fy para o modelo FSRCNN
        path = "./models/"+algoritmo+"/"+modelo+"/"+file

        sr.readModel(path)
        sr.setModel(setModel,fx)
        result = sr.upsample(img)
        
        resized = cv2.resize(img,dsize=None,fx=fx,fy=fy)

        print()
        psnr = utils.PSNR(resized, result)
        titulo = "Imagem Melhorada pelo Algoritmo:",algoritmo, "PSNR:",psnr
        print("Raça:",file,"- PSNR:", psnr)
        print()

        plt.figure(figsize=(8,6))
        plt.subplot(1,1,1)
        plt.title("Imagem Original")
        plt.imshow(img[:,:,::-1])

        plt.figure(figsize=(8,6))
        plt.subplot(1,1,1)
        plt.title(titulo)
        plt.imshow(result[:,:,::-1])

        plt.figure(figsize=(8,6))
        plt.subplot(1,1,1)
        plt.title("Imagem Melhorada pela biblioteca OpenCV (Redimensionamento comum)")
        plt.imshow(resized[:,:,::-1])
        plt.show()
        break

cap.release()
cv2.destroyAllWindows()