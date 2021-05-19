from View import Ui_Dialog
from Controller import Controller
import Helper
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QPixmap, QPicture, QPainter, QPen
import cv2
import numpy as np
from functools import partial
import os

def split_image(filename, fragment_height, fragment_width):
    img = cv2.imread(filename)
    print(img.shape)
    img_height, img_width, img_channels = img.shape
    n = img_height // fragment_height
    m = img_width // fragment_width
    img_array = []
    for i in range(n):
        for j in range(m):
            img_array.append(img[i*fragment_height:(i+1)*fragment_height, j*fragment_width:(j+1)*fragment_width])
    assert(len(img_array) == n*m)
    
    return np.array(img_array)
    
def draw_grid(height, width, vperiod, hperiod, device):
    painter = QPainter(device)
    painter.setPen(QPen(QtCore.Qt.black, 5, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap));
    x_coord = 0
    
    while x_coord <= width:
        x_coord += hperiod
        painter.drawLine(x_coord, 0, x_coord, height)
        
    y_coord = 0
    while y_coord <= height:
        y_coord += vperiod
        painter.drawLine(0, y_coord, width, y_coord)
    
# Custom QLabel widget
class QLabelImage(QtWidgets.QLabel): 
    coordinateChanged = QtCore.pyqtSignal(int, name='coordinateChanged')
    
    def set_pixmap_size(self, height, width, fragmentSize):
        self.pixmapHeight = height
        self.pixmapWidth = width
        self.fragmentSize = fragmentSize

    def mouseMoveEvent(self, event):
        if self.pixmapHeight == None:
            self.pixmapHeight = self.pixmap().height()
        if self.pixmapWidth == None:
            self.pixmapWidth = self.pixmap().width()
        
        cursorPos = event.localPos()
        imagePosX = cursorPos.x() * self.pixmapWidth / self.frameRect().width()
        imagePosY = cursorPos.y() * self.pixmapHeight / self.frameRect().height()
        n_column = imagePosX // self.fragmentSize
        n_row = imagePosY // self.fragmentSize
        n_columns = self.pixmapWidth // self.fragmentSize
        n_fragment = n_row * n_columns + n_column
        self.coordinateChanged.emit(n_fragment)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)

    ui.label = QLabelImage(ui.tab)
    ui.label.setGeometry(QtCore.QRect(20, 20, 801, 741))
    ui.label.setMouseTracking(True)
    ui.label.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
    ui.label.setFrameShape(QtWidgets.QFrame.Box)
    ui.label.setScaledContents(True)
    ui.label.setAlignment(QtCore.Qt.AlignCenter)
    ui.label.setObjectName("label_2")
    ui.label.coordinateChanged.connect(ui.frag_coord_label.setNum)
    
    PHOTO_PATH = "input.jpg"
    FRAGMENT_SIZE = 50
    
    # розділення зображення на вибрані фрагменти для кожного класу
    img_array = split_image(PHOTO_PATH, FRAGMENT_SIZE, FRAGMENT_SIZE)
    
    print(img_array.shape)
    # Обання текстури з 3х класів ліс, вода, земля
    img_class = np.array([img_array[79], img_array[154], img_array[147]])
    if not os.path.exists('temp'):
        os.mkdir('temp')
    for i in range(img_class.shape[0]):
        cv2.imwrite('temp/class{}.jpg'.format(i), img_class[i])
        
    # Показати oригінальне фото
    photo = QPixmap(PHOTO_PATH)
    Helper.showImage(photo, ui.label)
    ui.label.set_pixmap_size(photo.height(), photo.width(), FRAGMENT_SIZE)

    # Ініціалізація контроллера
    contrObject = Controller(PHOTO_PATH, ui.verticalLayoutWidget, ui.evLayout, ui.binLayout, ui.brightLayout, ui.verticalLayout, ui.checkBox_2, ui.comboBox, ui.gridLayoutWidget, ui.gridLayout_2, ui.delta_label, ui.label_2, (ui.red_checkBox, ui.green_checkBox, ui.blue_checkBox), ui.doubleSpinBox, ui.spinBox)
    
    # Set signal on button click to process the image and visualize result
    ui.pushButton.clicked.connect(partial(contrObject.process_data, img_array, img_class, FRAGMENT_SIZE))
    
    # Показати обрані текстури
    lbl_list = [ui.label_3, ui.label_4, ui.label_5]
    for i in range(img_class.shape[0]):
        Helper.showImage(QPixmap('temp/class'+str(i)+'.jpg'), lbl_list[i], contrObject.color_list[i + 1])
        ui.comboBox.addItem('Клас #' + str(i + 1))
    ui.comboBox.setCurrentIndex(0)

    # Нанести чорну сітку на зображення де кожна комірка це фрагмент зображення для класифікації
    photo_black_grid = QPixmap(PHOTO_PATH)
    draw_grid(photo_black_grid.height(), photo_black_grid.width(), FRAGMENT_SIZE, FRAGMENT_SIZE, photo_black_grid)
    ui.checkBox.clicked.connect(partial(Helper.show_image_checkbox, photo, photo_black_grid, ui.label, ui.checkBox))
    
    Dialog.show()
    sys.exit(app.exec_())