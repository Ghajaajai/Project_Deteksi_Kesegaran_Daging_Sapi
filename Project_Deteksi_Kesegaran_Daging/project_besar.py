import math
import sys
import cv2
import imutils
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt

# inisialisasi window
class showImage(QMainWindow):
    def __init__(self):
        super(showImage, self).__init__()
        loadUi('GUI.ui', self)
        self.Image = None

        self.actionOpen.triggered.connect(self.loadImage)
        self.btnCheck.clicked.connect(self.rgb_to_hsv)

    @pyqtSlot()
    def loadImage(self):
        image, filter = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\User\\', "Image Files(*.jpg)")
        self.Image = cv2.imread(image, 1)
        self.displayImage()

    # mengubah citra menjadi RGB ke HSV
    # def rgb_to_hsv(self, rgb_image):
    #     r, g, b = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]
    #
    #     # Normalize RGB values to the range [0, 1]
    #     r, g, b = r / 255.0, g / 255.0, b / 255.0
    #
    #     max_value = np.max(rgb_image, axis=2)
    #     min_value = np.min(rgb_image, axis=2)
    #
    #     # Compute the value component (V)
    #     v = max_value
    #
    #     # Compute the saturation component (S)
    #     s = (max_value - min_value) / max_value
    #
    #     # Compute the hue component (H)
    #     delta = max_value - min_value
    #     delta_zero_mask = delta == 0
    #     r_delta = np.where(delta_zero_mask, 0, (max_value - r) / delta)
    #     g_delta = np.where(delta_zero_mask, 0, (max_value - g) / delta)
    #     b_delta = np.where(delta_zero_mask, 0, (max_value - b) / delta)
    #
    #     h = np.zeros_like(max_value)
    #     h = np.where(max_value == r, 60 * (b_delta - g_delta) + 0, h)
    #     h = np.where(max_value == g, 60 * (r_delta - b_delta) + 120, h)
    #     h = np.where(max_value == b, 60 * (g_delta - r_delta) + 240, h)
    #     h = np.where(h < 0, h + 360, h)
    #
    #     hsv_image = np.stack((h, s, v), axis=2)
    #
    #     H, W = hsv_image.shape[:2]  # mengambil ukuran citra
    #     for i in range(H):
    #         for j in range(W):
    #             cv2.imshow(hsv_image)
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()

    # def rgb2grey(self):
    #     H, W = self.Image.shape[:2] # mengambil ukuran citra
    #     hsv = np.zeros((H, W), np.uint8) # membuat array kosong hsv dengan ukuran H dan W, dengan tipe data unsigned integer 8 bit (bilangan bulat positif)
    #     # looping untuk mengambil nilai piksel ke i (sumbu x) dan j (sumbu y)
    #     for i in range(H):
    #         for j in range(W):
    #             hsv[i, j] = np.clip(0.299 * self.Image[i, j, 0] + 0.587 * self.Image[i, j, 1] + 0.114 * self.Image[i, j, 2], 0, 255)
    #             print(hsv[i, j])
    #     # self.Image = hsv
    #     # self.displayImage(2)
    #             cv2.imshow('Test', hsv)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # HOG Descriptor
    # def HOGDescriptor(self):
    #     hog = cv2.HOGDescriptor()  # mengaktifkan fungsi cv2.HOGDescriptor kedalam variabel hog
    #     hog.setSVMDetector(
    #         cv2.HOGDescriptor_getDefaultPeopleDetector())  # GANTI UNTUK MENDETEKSI DAGING
    #     image, filter = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\User\\', "Image Files(*.jpg)")
    #
    #     img = self.Image
    #
    #     img = imutils.resize(img, width=min(400, img.shape[
    #         0]))  # meresize image dengan library imutlis dengan width minimal 400
    #     (regions, _) = hog.detectMultiScale(img, winStride=(4, 4), padding=(4, 4),
    #                                         scale=1.05)  # melakukan proses deteksi orang pada image
    #
    #     for (x, y, w, h) in regions:  # melakukan peloopingan sebanyak orang yang terdeteksi
    #         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255),
    #                       2)  # membuat bentuk persegi pada objek yang terdeteksi sebagai orang
    #
    #     cv2.imshow("image", img)  # menampilkannya dengan cv2.imshow
    #     cv2.waitKey()  # menunggu inputan user

    def displayImage(self):
        qformat = QImage.Format_Indexed8

        if len(self.Image.shape) == 3:
            if (self.Image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0], self.Image.strides[0], qformat)

        img = img.rgbSwapped()

        self.labelCitra.setPixmap(QPixmap.fromImage(img))
        self.labelCitra.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.labelCitra.setScaledContents(True)

app = QtWidgets.QApplication(sys.argv)
window = showImage()
window.setWindowTitle('Deteksi Kesegaran Daging')
window.show()
sys.exit(app.exec_())


# Alur program:
# Input Citra
# deteksi benda (daging) --> HOG Desc?
# setelah dideteksi bahwa citra adalah daging, deteksi warna :
# --> konversi warna rgb ke hsv
# --> ekstrak array hsv
# --> bandingkan dengan hasil klasifikasi (segar, agak segar, basi)