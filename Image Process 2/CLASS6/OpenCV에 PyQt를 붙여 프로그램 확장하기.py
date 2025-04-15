from PyQt5.QtWidgets import *
import sys
import cv2 as cv
import os

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
    sys.prefix, "Lib", "site-packages", "PyQt5", "Qt5", "plugins", "platforms"
)

class Video(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('비디오에서 프레임 수집')
        self.setGeometry(200, 200, 500, 100)

        videoButton = QPushButton('비디오 켜기', self)
        captureButton = QPushButton('프레임 잡기', self)
        saveButton = QPushButton('프레임 저장', self)
        quitButton = QPushButton('나가기', self)

        videoButton.setGeometry(10, 10, 100, 30)
        captureButton.setGeometry(110, 10, 100, 30)
        saveButton.setGeometry(210, 10, 100, 30)
        quitButton.setGeometry(310, 10, 100, 30)

        videoButton.clicked.connect(self.videoFunction)
        captureButton.clicked.connect(self.captureFunction)
        saveButton.clicked.connect(self.saveFunction)
        quitButton.clicked.connect(self.quitFunction)

    def videoFunction(self):
        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        if not self.cap.isOpened():
            self.close()
        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                break
            cv.imshow('video display', self.frame)
            cv.waitKey(1)

    def captureFunction(self):
        self.capturedFrame = self.frame
        cv.imshow('Captured Frame', self.capturedFrame)

    def saveFunction(self):
        fname = QFileDialog.getSaveFileName(self, '파일 저장', './')
        cv.imwrite(fname[0], self.capturedFrame)

    def quitFunction(self):
        self.cap.release()
        cv.destroyAllWindows()
        self.close()

app = QApplication(sys.argv)
win = Video()
win.show()
app.exec_()
