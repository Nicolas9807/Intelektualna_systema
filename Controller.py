import Helper
from System import ImageClassificator
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QPixmap, QPicture, QPainter, QPen, QBrush
import numpy as np
from functools import partial
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class Controller(QtCore.QObject):
    color_list = [QtCore.Qt.gray, QtCore.Qt.green, QtCore.Qt.blue, QtCore.Qt.red, QtCore.Qt.yellow]
    
    def __init__(self, image_path, layoutWidget, evLayout, binLayout, brightLayout, graphLayout, 
    show_recogn_checkBox, comboBox, optimalRadiusLayoutWidget, optimalRadiusLayout, optimalDeltaLabel, classificationLabel, channels_control_list, binerizationSpinBox, deltaSpinBox):
        self.image_path = image_path
        self.photo = QPixmap(image_path)
        self.layoutWidget = layoutWidget
        self.evLayout = evLayout
        self.binLayout = binLayout
        self.brightLayout = brightLayout
        self.graphLayout = graphLayout
        self.show_recogn_checkBox = show_recogn_checkBox
        self.ev_lbl_list = []
        self.bin_canvas_list = []
        self.bright_canvas_list = []
        self.canvas_list = []
        self.plot_list = []
        self.optimal_radius_list = []
        self.comboBox = comboBox
        self.classif = ImageClassificator()
        self.optimalRadiusLayout = optimalRadiusLayout
        self.optimalRadiusLayoutWidget = optimalRadiusLayoutWidget
        
        lbl1 = QtWidgets.QLabel(self.optimalRadiusLayoutWidget)
        lbl1.setFrameShape(QtWidgets.QFrame.Box)
        lbl1.setText("Клас №")
        optimalRadiusLayout.addWidget(lbl1, 0, 0, 1, 2)
        
        lbl2 = QtWidgets.QLabel(self.optimalRadiusLayoutWidget)
        lbl2.setFrameShape(QtWidgets.QFrame.Box)
        lbl2.setText("Оптимальний радіус = ")
        optimalRadiusLayout.addWidget(lbl2, 1, 0, 1, 2)
        
        self.optimalDeltaLabel = optimalDeltaLabel
        self.classificationLabel = classificationLabel
        self.channels_control_list = channels_control_list
        self.binerizationSpinBox = binerizationSpinBox
        self.deltaSpinBox = deltaSpinBox
        
    def draw_grid_class(self, image, vperiod, hperiod, class_predicted):
        painter_grid = QPainter(image)
        unique_class = list(set(class_predicted))
        class_counter = 0
        integer_height = (image.height() // vperiod) * vperiod
        integer_width = (image.width() // hperiod) * hperiod
        for y_coord in range(0, integer_height, vperiod):
            for x_coord in range(0, integer_width, hperiod):
                painter_grid.setPen(QPen(self.color_list[class_predicted[class_counter]], 5, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
                painter_grid.drawRect(x_coord, y_coord, hperiod, vperiod)
                class_counter += 1
        
    def set_label(self, lbl, layout, obj):
        lbl.setFrameShape(QtWidgets.QFrame.Box)
        if type(obj) == str:
            lbl.setText(obj)
        elif type(obj) == QPixmap:
            lbl.setPixmap(obj)
        layout.addWidget(lbl)
    
    def process_data(self, img_array, img_class, fragmentSize):
        channels = [i for i in range(len(self.channels_control_list)) if self.channels_control_list[i].isChecked()]
        if not channels:
            msgBox = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, "Некоректні дані", "Оберіть хоча б один канал")
            msgBox.exec()
            return
        n_base_class = self.comboBox.currentIndex()
        photo_predicted = QPixmap(self.image_path)
        system_is_fitted = self.classif.fit(img_class, n_base_class, channels, self.binerizationSpinBox.value(), self.deltaSpinBox.value())
        
        if system_is_fitted:
            self.clear_layout(self.optimalRadiusLayout, self.optimal_radius_list)
            for i in range(len(self.classif.rad_opt)):
                class_lbl = QtWidgets.QLabel(self.optimalRadiusLayoutWidget)
                class_lbl.setFrameShape(QtWidgets.QFrame.Box)
                class_lbl.setNum(i + 1)
                self.optimalRadiusLayout.addWidget(class_lbl, 0, 2 + i, 1, 1)
                
                radius_lbl = QtWidgets.QLabel(self.optimalRadiusLayoutWidget)
                radius_lbl.setFrameShape(QtWidgets.QFrame.Box)
                radius_lbl.setNum(self.classif.rad_opt[i])
                self.optimalRadiusLayout.addWidget(radius_lbl, 1, 2 + i, 1, 1)
                
                self.optimal_radius_list.extend([class_lbl, radius_lbl])
            
            class_predicted = self.classif.predict(img_array)
            self.draw_grid_class(photo_predicted, fragmentSize, fragmentSize, class_predicted)
            
            self.optimalDeltaLabel.setText("Оптимальна дельта = : +-{}".format(self.classif.delta_opt))
        else:
            msgBox = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Information, "Помилка розпізнавання", "Artificial system is not able to learn and predict any fragment!")
            msgBox.exec()
            
        self.show_EV(self.classif.etalonVectors)
        self.show_matrix(self.classif.binary_mat, self.binLayout, self.bin_canvas_list)
        self.show_matrix(self.classif.bright_mat, self.brightLayout, self.bright_canvas_list)
        
        self.clear_layout(self.graphLayout, self.canvas_list)
        self.clear_layout(self.graphLayout, self.plot_list)
        
        self.plot(self.classif.kfeRadius1, self.graphLayout, 0, self.canvas_list, "Залеж. КФЕ від радіуса і дельта = +-20", "радіус", "КФЕ")
        self.plot(self.classif.kfeRadius2, self.graphLayout, 1, self.canvas_list, "Залеж. КФЕ від радіусу опт. дельти", "радіус", "КФЕ")
        self.plot(self.classif.kfeDelta, self.graphLayout, 2, self.canvas_list, "Залеж. КФЕ від дельти", "дельта", "КФЕ")
        self.show_recogn_checkBox.clicked.connect(partial(Helper.show_image_checkbox, self.photo, photo_predicted, self.classificationLabel, self.show_recogn_checkBox))
    
    def clear_layout(self, layout, widget_list):
        for w in widget_list:
            layout.removeWidget(w)
        widget_list = []
        
    
    def show_EV(self, ev_list):
        self.clear_layout(self.evLayout, self.ev_lbl_list)
            
        self.pixm = []
        for index, ev in enumerate(ev_list):
            self.pixm.append(QPixmap(500, self.layoutWidget.width()))
            painter_grid = QPainter(self.pixm[-1])
            hstep = int(self.pixm[-1].width() / ev.shape[0])
            for i in range(ev.shape[0]):
                brush = QBrush(QtCore.Qt.yellow)
                if ev[i] == 0:
                    brush = QBrush(QtCore.Qt.black)
                elif ev[i] == 1:
                    brush = QBrush(QtCore.Qt.red)
                painter_grid.fillRect(i * hstep, 0, hstep, self.pixm[-1].height(), brush)
            
            ev_lbl_pix = QtWidgets.QLabel(self.layoutWidget)
            ev_lbl_pix.setFrameShape(QtWidgets.QFrame.Box)
            ev_lbl_pix.setObjectName("ev_lbl_pix"+str(index))
            ev_lbl_pix.setPixmap(self.pixm[-1])
            
            self.ev_lbl_list.append(ev_lbl_pix)
            self.evLayout.addWidget(ev_lbl_pix)
            
    def show_matrix(self, matrix, layout, canvas_list):
        self.clear_layout(layout, canvas_list)
        for i in range(matrix.shape[0]):
            canvas = FigureCanvas(Figure(figsize=(5, 3)))
            layout.addWidget(canvas)
            static_ax = canvas.figure.subplots()
            static_ax.imshow(matrix[i], cmap='gray')
            canvas_list.append(canvas)
    
    def plot(self, yx_vector, layout, number, canvas_list, title, xlabel, ylabel):
        figure = Figure(figsize=(10, 10))
        static_ax = figure.subplots()
        static_ax.set_title(title)
        for i in range(yx_vector[0].shape[0]):
            static_ax.plot(yx_vector[1], yx_vector[0][i], label='Клас '+str(i + 1))
        static_ax.legend()
        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)
        self.plot_list.append(canvas)
