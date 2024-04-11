import imageio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from math import cos, sin, pi, sqrt
import scipy.fftpack as TF
from skimage.transform import rotate


def imread(filename, greyscale=True):
    """ Transformation d'une image en tableau """
    if greyscale:
        pil_im = Image.open(filename).convert('L')  # converti en nuance de gris
    else:
        pil_im = Image.open(filename)
    return np.array(pil_im)


def agrandissement(im, n):
    x = im.shape[0]
    y = im.shape[1]
    img2 = np.zeros((x + 2 * n, y + 2 * n))
    img2[n: x + n, n: y + n] = im
    return img2


def affiche_image(im):
    """ Affiche une tableau numpy """
    plt.imshow(im)
    plt.show()


def saveAsPNG(im, name):
    """ Sauve un tableau numpy en png sous le nom 'name' """
    imageio.imwrite(name + '.png', im)


def radon(image, steps):
    projections = []
    dTheta = -180.0 / steps

    for i in range(steps):
        # on tourne l'image et on somme la verticale plutôt que de sommer sur des droites qu'on fait tourner
        projections.append(rotate(image, i * dTheta).sum(axis=0))  # on somme sur l'axe verticale

    return np.vstack(projections)  # on empile les projections


img = imread('radio_pied.png')
img2 = agrandissement(img, 100)
sino = radon(img2, 3000)

# %% Algorithme de reconstruction sans filtrage :

def reverse(im):
    xlen = im.shape[0]
    ylen = im.shape[1]
    resultat = np.zeros((ylen, ylen))
    dTheta = 180.0 / xlen
    for i in range(xlen):
        temp = np.tile(im[i], (ylen, 1))
        temp = rotate(temp, dTheta * i)
        resultat += temp
    return resultat


def Lambda(N, M):
    """ Filtre rampe """
    t = np.zeros((N, M))
    for k in range(N):
        t[k, :] = abs(pi * TF.fftfreq(M, 2 / N))
    return t


def Q(im, N, M):
    A = Lambda(N, M)  # Filtre
    B = TF.fft(im, axis=1)  # Transformée de Fourier de l'image (sur les lignes)
    C = A * B  # Multiplication de B par le filtre (on attenue les basse fréquence et on augmente les haute fréquence)
    return TF.ifft(C, axis=1)  # Transformée de Fourier inverse


def reverse2(im):  # avec filtrage
    xlen = im.shape[0]
    ylen = im.shape[1]
    im = Q(im, xlen, ylen).real  # On applique la fonction inverse au sinogramme filtré
    resultat = np.zeros((ylen, ylen))
    dTheta = 180.0 / xlen
    for i in range(xlen):
        temp = np.tile(im[i], (ylen, 1))
        temp = rotate(temp, dTheta * i)
        resultat += temp
    return resultat


a = reverse2(sino)
saveAsPNG(a, 'test6')
plt.imshow(a)
plt.show()

