# E01 - Aplikasi Deteksi Kesegaran Daging

import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt

class showImage(QMainWindow):
    def __init__(self):
        super(showImage, self).__init__()
        loadUi('GUI.ui', self)
        self.Image = None

        self.actionOpen.triggered.connect(self.loadImage)
        self.btnCheck.clicked.connect(self.detect)
        self.btncolorpick.clicked.connect(self.colorPickPic)

    def loadImage(self):
        image, filter = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\User\\', "Image Files(*.jpg)")
        self.Image = cv2.imread(image, 1)
        self.displayImage(1)

    def detect(self):
        # array untuk masking 1 berdasarkan nilai hsv
        lower = np.array([0, 35, 45])
        upper = np.array([179, 229, 240])

        # array threshold nilai hsv untuk daging kualitas yang bagus
        lower_normal = np.array([0, 106, 154])
        upper_normal = np.array([10, 255, 246])

        # array threshold nilai hsv untuk kualitas daging yang jelek
        lower_jelek = np.array([0, 0, 0])
        upper_jelek = np.array([10, 110, 134])

        # Read Image
        img = self.Image
        self.imgawal = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Konvolusi dengan kernel gaussian
        img = cv2.GaussianBlur(img, (5, 5), 0) # jabarkan
        self.imggauss = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Filtering dengan masking 1
        mask = cv2.inRange(result, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        result = cv2.dilate(mask, kernel)

        # membuat contours
        contours, hierarchy = cv2.findContours(result.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        self.imgmask_background = result

        # menentukan kualitas masing-masing daging yang terdeteksi
        if len(contours) != 0:
            for (i, c) in enumerate(contours):
                area = cv2.contourArea(c)
                print("area : ", area)
                if area > 1000:
                    cv2.drawContours(img, c, -1, (255, 255, 0), 8)  # menggambar contours hasil mask
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)
                    croppedk = img[y:y + h, x:x + w]

                    hsv = cv2.cvtColor(croppedk, cv2.COLOR_BGR2HSV)

                    # deteksi kualitas daging
                    # warna normal
                    normalmask = cv2.inRange(hsv, lower_normal, upper_normal)

                    cv2.imshow('mask warna normal', normalmask)
                    cnt_n = 0
                    for n in normalmask:
                        cnt_n = cnt_n + list(n).count(255)
                    print("nilai mask normal : ", cnt_n)

                    # warna jelek
                    badmask = cv2.inRange(hsv, lower_jelek, upper_jelek)

                    cv2.imshow('mask warna busuk', badmask)
                    cnt_j = 0
                    for j in badmask:
                        cnt_j = cnt_j + list(j).count(255)
                    print("nilai mask jelek : ", cnt_j)

                    # Penamaan klasifikasi
                    # segar
                    if cnt_n > cnt_j:
                        cv2.putText(img, 'Daging masih segar', (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        print("Terdeteksi : daging bagus")
                    # busuk
                    if cnt_j > cnt_n:
                        cv2.putText(img, 'Daging sudah busuk', (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        print("Terdeteksi : daging busuk")
            print("-------------------------------")
        else:
            print("Tidak ada objek yang terdeteksi")

        # Tampilkan hasil
        self.Image = img
        self.imgakhir = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.displayImage(2)
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Tampilkan histogram dari hasil
        h, s, v = img2[:, :, 0], img2[:, :, 1], img2[:, :, 2]
        hist_h = cv2.calcHist([h], [0], None, [256], [0, 256])
        hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
        hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])
        plt.plot(hist_h, color='r', label="h")
        plt.plot(hist_s, color='g', label="s")
        plt.plot(hist_v, color='b', label="v")
        plt.legend()
        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def displayImage(self, windows=1):
        qformat = QImage.Format_Indexed8

        if len(self.Image.shape) == 3:
            if (self.Image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888

            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0],
                     self.Image.strides[0], qformat)

        img = img.rgbSwapped()
        if windows == 1:
            self.labelCitra.setPixmap(QPixmap.fromImage(img))

            self.labelCitra.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

            self.labelCitra.setScaledContents(True)
        if windows == 2:
            self.labelCitra_2.setPixmap(QPixmap.fromImage(img))

            self.labelCitra_2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

            self.labelCitra_2.setScaledContents(True)

    def colorPickPic(self):
        def nothing(x):
            pass

        img = self.Image
        cv2.namedWindow("Trackbar")
        cv2.createTrackbar("Lower H", "Trackbar", 0, 179, nothing)
        cv2.createTrackbar("Lower S", "Trackbar", 0, 255, nothing)
        cv2.createTrackbar("Lower V", "Trackbar", 0, 255, nothing)
        cv2.createTrackbar("Upper H", "Trackbar", 179, 179, nothing)
        cv2.createTrackbar("Upper S", "Trackbar", 255, 255, nothing)
        cv2.createTrackbar("Upper V", "Trackbar", 255, 255, nothing)

        while True:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            LH = cv2.getTrackbarPos("Lower H", "Trackbar")
            LS = cv2.getTrackbarPos("Lower S", "Trackbar")
            LV = cv2.getTrackbarPos("Lower V", "Trackbar")
            UH = cv2.getTrackbarPos("Upper H", "Trackbar")
            US = cv2.getTrackbarPos("Upper S", "Trackbar")
            UV = cv2.getTrackbarPos("Upper V", "Trackbar")
            lower_color = np.array([LH, LS, LV])
            upper_color = np.array([UH, US, UV])
            mask = cv2.inRange(hsv, lower_color, upper_color)
            result = cv2.bitwise_and(img, img, mask=mask)
            cv2.imshow("Frame", img)
            cv2.imshow("Mask", mask)
            cv2.imshow("Hasil", result)
            key = cv2.waitKey(1)
            if key == 27:  # tekan esc untuk stop
                break
        cv2.destroyAllWindows()

app = QtWidgets.QApplication(sys.argv)
window = showImage()
window.setWindowTitle('Deteksi Kesegaran Daging')
window.show()
sys.exit(app.exec_())