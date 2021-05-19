from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap, QPicture, QPainter, QPen
    
def show_image_checkbox(image, image_pred, lbl, checkBox):
    if checkBox.isChecked():
        showImage(image_pred, lbl)
    else:
        showImage(image, lbl)

def showImage(pix, lbl, border_color = None):
    if border_color != None:
        painter = QPainter(pix)
        painter.setPen(QPen(border_color, 5, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        painter.drawRect(0,0,lbl.frameRect().width(),lbl.frameRect().height())
    
    if type(pix) == QPixmap:
        lbl.setPixmap(pix.scaled(lbl.frameRect().height(), 
            lbl.frameRect().width(), QtCore.Qt.KeepAspectRatio))
    elif type(pix) == QPicture:
        lbl.setPicture(pix)