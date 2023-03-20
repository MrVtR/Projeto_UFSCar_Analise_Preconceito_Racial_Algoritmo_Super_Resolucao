import cv2
import matplotlib.pyplot as plt
import utils

imageTestFiles=['asiatico','branco','indiano','latino','negro','oriente']
CNNFolders = ['FSRCNN','TF-ESPCN','TF-LapSRN']
modelsFolders = ['asian','black','indian','latino','middle','mixed','white']

for file in imageTestFiles:
  img = cv2.imread("images/"+file+".jpg")

  for model in modelsFolders:
    print("Modelo:",model)
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    #Path para o modelo FSRCNN
    path = "./models/FSRCNN/"+model+"/FSRCNN_x2.pb"

    #Path para os modelos TF-ESPCN
    # path = './models/TF-ESPCN'+model+'frozen_ESPCN_graph_x4.pb'

    #Path para os modelos TF-Lap-SRN
    # path = './models/TF-LapSRN'+'LapSRN_x2.pb'
    sr.readModel(path)
    sr.setModel("fsrcnn",2)
    result = sr.upsample(img)
    
    # Resized image
    resized = cv2.resize(img,dsize=None,fx=2,fy=2)

    print("Raça:",file,"- PSNR:", utils.PSNR(resized, result))
    print()

  print("--------------------------------------------------------")


