import argparse
import numpy as np
from PIL import Image

'''
normal norm/process 230
'''

def normalizeStaining(img, saveFile=None, Io=240, alpha=1, beta=0.15):

    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])

    maxCRef = np.array([1.9705, 1.0308])

    h, w, c = img.shape
    img = img.reshape((-1, 3))
    OD = -np.log((img.astype(np.float) + 1) / Io)
    ODhat = OD[~np.any(OD < beta, axis=1)]

    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

    # eigvecs *= -1
    That = ODhat.dot(eigvecs[:, 1:3])

    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    Y = np.reshape(OD, (-1, 3)).T

    C = np.linalg.lstsq(HE, Y, rcond=None)[0]

    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])

    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
    H[H > 255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))
    E[E > 255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)

    if saveFile is not None:
        Image.fromarray(Inorm).save(saveFile + '.png')
        Image.fromarray(H).save(saveFile + '_H.png')
        Image.fromarray(E).save(saveFile + '_E.png')
    return Inorm, H, E

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--imageFile', type=str, default='example1.tif', help='RGB image file')
#     parser.add_argument('--saveFile', type=str, default='output', help='save file')
#     parser.add_argument('--Io', type=int, default=240)
#     parser.add_argument('--alpha', type=float, default=1)
#     parser.add_argument('--beta', type=float, default=0.15)
#     args = parser.parse_args()
#
#     img = np.array(Image.open(args.imageFile))
#
#     normalizeStaining(img=img,
#                       saveFile=args.saveFile,
#                       Io=args.Io,
#                       alpha=args.alpha,
#                       beta=args.beta)