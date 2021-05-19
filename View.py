from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1400, 843)
        Dialog.setToolTipDuration(2)
        self.tabWidget = QtWidgets.QTabWidget(Dialog)
        self.tabWidget.setGeometry(QtCore.QRect(20, 20, 1071, 821))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.pushButton = QtWidgets.QPushButton(self.tab)
        self.pushButton.setGeometry(QtCore.QRect(880, 700, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.n_frags_label = QtWidgets.QLabel(self.tab)
        self.n_frags_label.setGeometry(QtCore.QRect(870, 640, 161, 51))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.n_frags_label.setFont(font)
        self.n_frags_label.setObjectName("n_frags_label")
        self.label_4 = QtWidgets.QLabel(self.tab)
        self.label_4.setGeometry(QtCore.QRect(940, 60, 51, 51))
        self.label_4.setObjectName("label_4")
        self.label_3 = QtWidgets.QLabel(self.tab)
        self.label_3.setGeometry(QtCore.QRect(850, 60, 51, 51))
        self.label_3.setObjectName("label_3")
        self.label_5 = QtWidgets.QLabel(self.tab)
        self.label_5.setGeometry(QtCore.QRect(850, 150, 51, 51))
        self.label_5.setObjectName("label_5")
        self.frag_coord_label = QtWidgets.QLabel(self.tab)
        self.frag_coord_label.setGeometry(QtCore.QRect(890, 500, 101, 20))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.frag_coord_label.setFont(font)
        self.frag_coord_label.setFrameShape(QtWidgets.QFrame.Box)
        self.frag_coord_label.setScaledContents(True)
        self.frag_coord_label.setAlignment(QtCore.Qt.AlignCenter)
        self.frag_coord_label.setObjectName("frag_coord_label")
        self.image_info_label = QtWidgets.QLabel(self.tab)
        self.image_info_label.setGeometry(QtCore.QRect(890, 560, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.image_info_label.setFont(font)
        self.image_info_label.setObjectName("image_info_label")
        self.image_size_label = QtWidgets.QLabel(self.tab)
        self.image_size_label.setGeometry(QtCore.QRect(870, 580, 151, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.image_size_label.setFont(font)
        self.image_size_label.setObjectName("image_size_label")
        self.comboBox = QtWidgets.QComboBox(self.tab)
        self.comboBox.setGeometry(QtCore.QRect(940, 290, 69, 22))
        self.comboBox.setObjectName("comboBox")
        self.checkBox = QtWidgets.QCheckBox(self.tab)
        self.checkBox.setGeometry(QtCore.QRect(860, 450, 161, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.checkBox.setFont(font)
        self.checkBox.setObjectName("checkBox")
        self.label = QtWidgets.QLabel(self.tab)
        self.label.setGeometry(QtCore.QRect(850, 290, 71, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_7 = QtWidgets.QLabel(self.tab)
        self.label_7.setGeometry(QtCore.QRect(850, 30, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.tab)
        self.label_8.setGeometry(QtCore.QRect(940, 30, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.tab)
        self.label_9.setGeometry(QtCore.QRect(850, 120, 51, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(self.tab)
        self.doubleSpinBox.setGeometry(QtCore.QRect(940, 330, 62, 22))
        self.doubleSpinBox.setDecimals(3)
        self.doubleSpinBox.setMaximum(1.0)
        self.doubleSpinBox.setSingleStep(0.01)
        self.doubleSpinBox.setProperty("value", 0.1)
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        self.label_14 = QtWidgets.QLabel(self.tab)
        self.label_14.setGeometry(QtCore.QRect(850, 330, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.spinBox = QtWidgets.QSpinBox(self.tab)
        self.spinBox.setGeometry(QtCore.QRect(940, 389, 61, 21))
        self.spinBox.setMinimum(0)
        self.spinBox.setMaximum(50)
        self.spinBox.setProperty("value", 20)
        self.spinBox.setObjectName("spinBox")
        self.label_15 = QtWidgets.QLabel(self.tab)
        self.label_15.setGeometry(QtCore.QRect(850, 380, 61, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.tab)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(850, 210, 160, 51))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.red_checkBox = QtWidgets.QCheckBox(self.gridLayoutWidget_2)
        self.red_checkBox.setChecked(True)
        self.red_checkBox.setObjectName("red_checkBox")
        self.gridLayout.addWidget(self.red_checkBox, 1, 0, 1, 1)
        self.green_checkBox = QtWidgets.QCheckBox(self.gridLayoutWidget_2)
        self.green_checkBox.setChecked(True)
        self.green_checkBox.setObjectName("green_checkBox")
        self.gridLayout.addWidget(self.green_checkBox, 1, 1, 1, 1)
        self.blue_checkBox = QtWidgets.QCheckBox(self.gridLayoutWidget_2)
        self.blue_checkBox.setChecked(True)
        self.blue_checkBox.setObjectName("blue_checkBox")
        self.gridLayout.addWidget(self.blue_checkBox, 1, 2, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_16.setAlignment(QtCore.Qt.AlignCenter)
        self.label_16.setObjectName("label_16")
        self.gridLayout.addWidget(self.label_16, 0, 0, 1, 3)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.tab_2)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(20, 510, 571, 151))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.evLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.evLayout.setContentsMargins(0, 0, 0, 0)
        self.evLayout.setObjectName("evLayout")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.tab_2)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(20, 260, 581, 171))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.binLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.binLayout.setContentsMargins(0, 0, 0, 0)
        self.binLayout.setObjectName("binLayout")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.tab_2)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(30, 50, 571, 151))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.brightLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.brightLayout.setContentsMargins(0, 0, 0, 0)
        self.brightLayout.setObjectName("brightLayout")
        self.delta_label = QtWidgets.QLabel(self.tab_2)
        self.delta_label.setGeometry(QtCore.QRect(20, 680, 201, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.delta_label.setFont(font)
        self.delta_label.setObjectName("delta_label")
        self.label_11 = QtWidgets.QLabel(self.tab_2)
        self.label_11.setGeometry(QtCore.QRect(240, 20, 141, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.tab_2)
        self.label_12.setGeometry(QtCore.QRect(240, 220, 111, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.tab_2)
        self.label_13.setGeometry(QtCore.QRect(240, 460, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.gridLayoutWidget = QtWidgets.QWidget(self.tab_2)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(260, 680, 331, 51))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setSpacing(0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.tab_2)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(660, 50, 381, 731))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.checkBox_2 = QtWidgets.QCheckBox(self.tab_3)
        self.checkBox_2.setGeometry(QtCore.QRect(860, 380, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.checkBox_2.setFont(font)
        self.checkBox_2.setObjectName("checkBox_2")
        self.label_2 = QtWidgets.QLabel(self.tab_3)
        self.label_2.setGeometry(QtCore.QRect(20, 20, 801, 741))
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap("input.jpg"))
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")
        self.tabWidget.addTab(self.tab_3, "")

        self.retranslateUi(Dialog)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Столяр Микола IEIT"))
        self.pushButton.setText(_translate("Dialog", "Пуск"))
        self.label_4.setText(_translate("Dialog", "TextLabel"))
        self.label_3.setText(_translate("Dialog", "TextLabel"))
        self.label_5.setText(_translate("Dialog", "TextLabel"))
        self.frag_coord_label.setText(_translate("Dialog", "0"))
        self.checkBox.setText(_translate("Dialog", "Накласти сітку"))
        self.label.setText(_translate("Dialog", " Баз. клас:"))
        self.label_7.setText(_translate("Dialog", "Клас #1"))
        self.label_8.setText(_translate("Dialog", "Клас #2"))
        self.label_9.setText(_translate("Dialog", "Клас #3"))
        self.label_14.setText(_translate("Dialog", "Поріг\n" "бінаризації"))
        self.label_15.setText(_translate("Dialog", "Попереднє\n" "дельта"))
        self.red_checkBox.setText(_translate("Dialog", "Черв."))
        self.green_checkBox.setText(_translate("Dialog", "Зел."))
        self.blue_checkBox.setText(_translate("Dialog", "Синій"))
        self.label_16.setText(_translate("Dialog", "Вибрати колір каналу"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Dialog", "Ініціалізація"))
        self.delta_label.setText(_translate("Dialog", "Оптимальні значення дельта: "))
        self.label_11.setText(_translate("Dialog", "Навчальні матриці"))
        self.label_12.setText(_translate("Dialog", "Бінарні матриці"))
        self.label_13.setText(_translate("Dialog", "Ет. вектори"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("Dialog", "Навчання"))
        self.checkBox_2.setText(_translate("Dialog", " Показати\n" " результат"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("Dialog", "Результат екзамену"))