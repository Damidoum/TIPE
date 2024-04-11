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


def taille(im):
    return img.shape


def affiche_image(im):
    """ Affiche une tableau numpy """
    plt.imshow(im)
    plt.show()


def saveAsPNG(im, name):
    """ Sauve un tableau numpy en png sous le nom 'name' """
    imageio.imwrite(name + '.png', im)


# Importation de l'image
img = imread('original.png')


# %% Quelques fonctions qui vont servir :

def dansImage(im, i, j):
    """ verifie qu'un pixel est dans l'image """
    if i >= 0 and i < x and j >= 0 and j < y:
        return True
    else:
        return False


def color(im, i, j, c):
    """modifie la couleur de certains pixels de l'image"""
    if dansImage(im, i, j):
        im[i, j] = c


# %% Premiere methode (mauvais resultats)

def proj_graph(im, m, p, horizontale=False):
    """ Modifie la couleur des pixels sur le long d'une droite de coef directeur m et d'ordonnee a l'origine p """
    if horizontale:
        for i in range(0, x):
            color(img, i, p, 255)
        horizontale = False
    else:
        for i in range(-d1, d1):
            j = round(m * i + p)
            color(img, i, j, 255)
    plt.figure(dpi=300)
    plt.imshow(img)
    plt.show()


def proj(im, m, p, horizontale=False):
    """ Projection selon la droite de coef directeur m et d'ordonnee a l'origine p"""
    tot = 0
    if horizontale:
        for i in range(0, x):
            tot += im[i, p]
    else:
        # on somme les valeurs des pixels qui sont sur la droite
        for i in range(-d1, d1):
            j = round(m * i + p)
            if dansImage(im, i, j):
                tot += im[i, j]
    return tot


def coupe(img, m, pas=50, horizontale=False):
    """ Fait une projection selon une direction """
    lst = []
    a = round(-m * x)
    p = round((-a + x) / pas)
    # met dans une liste les coupes d'une meme direction
    for k in range(a, x, p):
        lst.append(proj(img, m, k, horizontale))
    while len(lst) < pas + 15:  # 15 a ajuster pour qu'il n'y ai pas de depassement du tableau
        lst.append(0)
    return lst


def coupe_graph(im, m, pas=50):
    """ Affiche les droites de projections pour une direction """
    a = round(-m * x)
    for k in np.linspace(a, x, pas):
        for i in range(-d1, d1):
            j = round(m * i + k)
            color(im, i, j, 255)
    plt.figure(dpi=300)
    plt.imshow(img)
    plt.show()


def sinogramme(im, pas=10):
    """ Retourne le sinogramme """
    l = np.linspace(10, 0, pas)
    projections = []
    for k in l:
        projections.append(coupe(img, k))
    return np.vstack(projections)


# %% Deuxieme methode avec des coordonnees polaires

def coupe_graph_polaire(img, theta, s):
    """ Affiche une droite de projection """
    for k in range(-int(1.4 * x), int(1.4 * y)):
        i, j = round(s * cos(theta) - k * sin(theta)), round(s * sin(theta) + k * cos(theta))
        color(img, i, j, 255)

    plt.figure(dpi=300)
    plt.imshow(img)
    plt.show()


def coupe_polaire(im, theta, s):
    """ Projection selon une droite de parametre theta, s """
    tot = 0
    for k in range(round(-1.4 * x), round(1.4 * x)):
        i, j = round(s * cos(theta) - k * sin(theta) + x / 2), round(s * sin(theta) + k * cos(theta) + x / 2)
        if i >= 0 and i < x and j >= 0 and j < y:
            tot += im[j, i]
    return int(tot)


def sinogrammeV2(im, M=50):
    """ Realisation du sinogramme """
    projections = []
    for m in range(0, M):
        lst = []
        for k in range(-int(x / 2), int(x / 2) + 1):
            lst.append(coupe_polaire(im, -m * pi / M, k))
        projections.append(lst)
    return np.vstack(projections)  # on empile les projections


# %% On propose une derniere methode avec la fonction rotate 

def radon(image, steps):
    """ Meilleure methode pour faire le sinogramme """
    projections = []
    dTheta = -180.0 / steps

    for i in range(steps):
        # on tourne l'image et on somme la verticale plutot que de sommer sur des droites qu'on fait tourner
        projections.append(rotate(image, i * dTheta).sum(axis=0))  # on somme sur l'axe verticale

    return np.vstack(projections)  # on empile les projections


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


# %% Avec le filtrage :

def Lambda(N, M):
    """ Filtre rampe """
    t = np.zeros((N, M))
    for k in range(N):
        t[k, :] = abs(pi * TF.fftfreq(M, 2 / N))
    return t


def Q(im, N, M):
    A = Lambda(N, M)  # Filtre
    B = TF.fft(im, axis=1)  # Transformee de Fourier de l'image (sur les lignes)
    C = A * B  # Multiplication de B par le filtre (on attenue les basse frequence et on augmente les haute frequence)
    return TF.ifft(C, axis=1)  # Transformee de Fourier inverse


def reverse2(im):  # avec filtrage
    xlen = im.shape[0]
    ylen = im.shape[1]
    im = Q(im, xlen, ylen).real  # On applique la fonction inverse au sinogramme filtre
    resultat = np.zeros((ylen, ylen))
    dTheta = 180.0 / xlen
    for i in range(xlen):
        temp = np.tile(im[i], (ylen, 1))
        temp = rotate(temp, dTheta * i)
        resultat += temp
    return resultat



