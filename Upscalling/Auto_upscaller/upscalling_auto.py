import cv2
import matplotlib.pyplot as plt
import utils

# files = ['branco','black','asia','indian','latino','oriente']

files=['black']

for file in files:
  img = cv2.imread(file+".jpg")

  sr = cv2.dnn_superres.DnnSuperResImpl_create()
  path = "asian_nchw_frozen_ESPCN_graph_x4.pb"
  sr.readModel(path)
  sr.setModel("espcn",4)
  result = sr.upsample(img)
  
  # Resized image
  resized = cv2.resize(img,dsize=None,fx=4,fy=4)

  print("Raça:",file,"- PSNR of ESPCN generated image em relação a imagem original em Resize: ", utils.PSNR(resized, result))
  print()


  plt.figure(figsize=(6,5))
  plt.subplot(1,1,1)
  plt.imshow(img[:,:,::-1])

  plt.figure(figsize=(6,5))
  plt.subplot(1,1,1)
  plt.imshow(result[:,:,::-1])

  plt.figure(figsize=(6,5))
  plt.subplot(1,1,1)
  plt.imshow(resized[:,:,::-1])
  plt.show()