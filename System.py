import numpy as np
import math

def hemming_distance(v1, v2):
    assert v1.shape == v2.shape, "{}, {}".format(v1.shape, v2.shape)
    return np.sum(v1 ^ v2)

def entropy(x1, x2):
    tolerance = 1e-9
    r = x1 / (x1 + x2)
    res = np.zeros(r.shape)
    for i in range(r.shape[0]):
        if r[i] > tolerance:
            res[i] = r[i] * math.log(r[i], 2)
    return res

def shennon(d1, alpha, beta, d2):
    return 1 + 0.5 * (entropy(alpha, d2) + entropy(d1, beta) + entropy(beta, d1) + entropy(d2, alpha))
    
def Kulbak(alpha, beta):
    small_number = 0.000001
    kulbak = 0.5 * (np.log2((2 - (alpha + beta + small_number)) / (alpha + beta + small_number))) * (1 - (alpha + beta))
    return kulbak


class ImageClassificator:
    def __init__ (self):
        self.etalonVectors = []
        self.kfeRadius1 = [np.array([]), np.array([])]
        self.kfeRadius2 = [np.array([]), np.array([])]
        self.kfeDelta = [np.array([]), np.array([])]
    
    def calcBrightnessMatrix(self, imgArray):
        if len(imgArray.shape) == 4:
            imgFilter = np.zeros((imgArray.shape[0], imgArray.shape[1], imgArray.shape[2]))
            for ch in self.channels:
                imgFilter += imgArray[:, :, :, ch]
            obuch_matr = (imgFilter/len(self.channels)).round()
            return obuch_matr
        else:
            return imgArray
        
    def calcBinaryMatrix(self, bright_matrix, ndk, vdk):
        binaryMat = np.zeros(bright_matrix.shape, dtype=int)
        for i in range(bright_matrix.shape[1]):
            binaryMat[:,i] = (bright_matrix[:,i] >= ndk) & (bright_matrix[:,i] <= vdk)
        return binaryMat
        
    def fit(self, imageArray, nBaseClass, channels, binarizationThreshold, predefinedDelta):
        self.nClasses = imageArray.shape[0]
        self.nRealiz = imageArray.shape[1]
        hyperRadii = np.arange(1, self.nRealiz+1)
        self.kfeDelta[0] = np.zeros((self.nClasses, self.nRealiz - 1))
        self.kfeDelta[1] = np.arange(1, self.nRealiz)
        isLearnedSuccessfully = False
        zeroDists = self.nClasses
        self.delta_opt = None

        kfeMEAN = -1
        self.channels = channels
        for delta in np.arange(1, self.nRealiz, 1):
            bright_mat = self.calcBrightnessMatrix(imageArray)

            sr_etal = bright_mat[nBaseClass].mean(0)
            vdk = sr_etal + delta
            ndk = sr_etal - delta
            
            binary_mat = self.calcBinaryMatrix(bright_mat, ndk, vdk)

            etalonVectors = np.array(binary_mat.mean(1) >= binarizationThreshold, int)
            # Hemming distance between etalon vectors and each realizations
            pairClasses = np.full(self.nClasses, -1)
            cl_dis = []
            zr_ds = 0
            for i in range(self.nClasses):
                class_center_distance = self.nRealiz + 1
                for j in range(self.nClasses):
                    if i == j:
                        continue
                    dist = hemming_distance(etalonVectors[i], etalonVectors[j])
                    if class_center_distance > dist:
                        class_center_distance = dist
                        pairClasses[i] = j
                cl_dis.append(class_center_distance)
                if class_center_distance == 0:
                    zr_ds += 1
            if zeroDists > zr_ds:
                # If prediction fails we always can get binary, brightness matrices and etalon vector
                zeroDists = zr_ds
                self.etalonVectors = etalonVectors
                self.binary_mat = binary_mat
                self.bright_mat = bright_mat
            
            EV_size = etalonVectors.shape[1]
            sk1 = np.zeros((self.nClasses, EV_size))
            sk2 = np.zeros((self.nClasses, EV_size))
            k1 = np.zeros((self.nClasses, EV_size))
            k2 = np.zeros((self.nClasses, EV_size))
            d1 = np.zeros((self.nClasses, EV_size))
            d2 = np.zeros((self.nClasses, EV_size))
            beta = np.zeros((self.nClasses, EV_size))
            alpha = np.zeros((self.nClasses, EV_size))
            e = np.zeros((self.nClasses, EV_size))
            hyperRadii = np.arange(1, etalonVectors.shape[1]+1)
            optimal_radius = np.zeros(self.nClasses)
            isGoodRadius = True
            for c in range(self.nClasses):
                for i in range(self.nRealiz):
                    sk1[c, i] = hemming_distance(etalonVectors[c], binary_mat[c, i])
                    sk2[c, i] = hemming_distance(etalonVectors[c], binary_mat[pairClasses[c], i])

                for i in range(self.nRealiz):
                    k1[c, i] = np.sum(sk1[c] <= hyperRadii[i])
                    k2[c, i] = np.sum(sk2[c] <= hyperRadii[i])

                # For Shennon, Kulbak
                n = self.nRealiz
                d1[c] = k1[c] / n
                beta[c] = k2[c] / n
                alpha[c] = 1 - d1[c]
                d2[c] = 1 - beta[c]
                e[c] = Kulbak(alpha[c], beta[c])

                good_radius_indices = (np.argwhere((d1[c] > 0.5) & (d2[c] > 0.5) & (hyperRadii <= class_center_distance)))
                if (good_radius_indices.shape[0] * good_radius_indices.shape[1]) == 0:
                    isGoodRadius = False
                    continue
                kfe_max = e[c][good_radius_indices].max()
                self.kfeDelta[0][c, delta - 1] = kfe_max
                optimal_radius[c] = hyperRadii[np.argwhere(kfe_max == e[c])[-1][0]]
                
            if predefinedDelta == delta:
                self.kfeRadius1 = [e, hyperRadii]
            if isGoodRadius:
                kfeMean = e[c][good_radius_indices].mean()
                if (kfeMEAN < kfeMean):
                    isLearnedSuccessfully = True
                    kfeMEAN = kfeMean
                    self.delta_opt = delta
                    self.etalonVectors = etalonVectors
                    self.rad_opt = optimal_radius
                    self.ndk = ndk
                    self.vdk = vdk
                    self.d1 = d1
                    self.d2 = d2
                    self.alpha = alpha
                    self.beta = beta
                    self.binary_mat = binary_mat
                    self.bright_mat = bright_mat
                    self.kfeRadius2 = [e, hyperRadii]
        return isLearnedSuccessfully

    def predict(self, imageArray):
        class_predicted = []
        for img_test in imageArray:
            bright_test = self.calcBrightnessMatrix(img_test.reshape((1, *img_test.shape)))
            binary_mat_test = self.calcBinaryMatrix(bright_test, self.ndk, self.vdk)[0]

            # Calculate the distance between the standard and the implementations of the corresponding image
            sk_test = np.zeros((self.nClasses, self.nRealiz))
            mu = np.zeros((self.nClasses, self.nRealiz))
            for c in range(self.nClasses):
                for i in range(self.nRealiz):
                    sk_test[c, i] = hemming_distance(self.etalonVectors[c], binary_mat_test[i])
                mu[c] = 1 - sk_test[c] / self.rad_opt[c]
            muAverage = mu.mean(1)
            
            # Whom belong
            sred_max = -1.0
            class_detected = 0 #None
            for c in range(self.nClasses):
                if (sred_max < muAverage[c]) and (muAverage[c] > 0):
                    sred_max = muAverage[c]
                    class_detected = c + 1
            class_predicted.append(class_detected)
            
        return class_predicted
