import cv2
import numpy as np

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def binarizar_imagem(caminho_arquivo):
	img = cv2.imread(caminho_arquivo)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	suave = cv2.GaussianBlur(img, (7, 7), 0)  # aplica blur
	(T, bin) = cv2.threshold(suave, 160, 255, cv2.THRESH_BINARY)
	(T, binI) = cv2.threshold(suave, 160, 255, cv2.THRESH_BINARY_INV)
	resultado = np.vstack([
		np.hstack([suave, bin]),
		np.hstack([binI, cv2.bitwise_and(img, img, mask = binI)])
		])

	return resultado

imagem_binarizada = binarizar_imagem('gato.jpg')
resize = ResizeWithAspectRatio(imagem_binarizada, width=1280)
cv2.imshow('Binarização da imagem', resize)
cv2.waitKey(0)
