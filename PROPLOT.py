import numpy as np
import pyqtgraph as pg
from collections import deque
from PyQt5 import QtWidgets, QtCore
from pyqtgraph import PlotWidget, plot
import time
import serial 
import signal
import sys
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.cluster import KMeans


#  on click on widget get x,y coordinates
def SignalHandler(sig, frame):
    print("SignalHandler")

    app.quit()
    
    sys.exit(0)



    

class MainWindow(QtWidgets.QMainWindow):
   
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.graphWidget = pg.PlotWidget()
        
        self.trackingWidget = pg.PlotWidget()

        self.gain = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.gain.setMinimum(0)
        self.gain.setMaximum(100)
        self.gain.setValue(30)

        self.gain.setTickInterval(1)
     
        self.range = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.range.setMinimum(0)
        self.range.setMaximum(500)
        self.range.setValue(0)
        self.range.setTickInterval(1)
        self.range.valueChanged.connect(self.rangeChanged)


        self.treshold = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.treshold.setMinimum(0)
        self.treshold.setMaximum(500)
        self.treshold.setValue(0)
        self.treshold.setTickInterval(1)
        self.treshold.valueChanged.connect(self.tresholdChanged)


        self.gain_label = QtWidgets.QLabel()
        self.gain_label.setText("Gain: " + str(self.gain.value()))
        
        self.treshold_label = QtWidgets.QLabel()
        self.treshold_label.setText("Treshold: " + str(self.treshold.value()))

        self.range_label = QtWidgets.QLabel()
        self.range_label.setText("Range: " + str(self.treshold.value()))

        self.trackingWidget.setBackground('w')





        
        self.graphWidget.setBackground('w')
        self.graphWidget.scene().sigMouseClicked.connect(self.mouseClicked)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.graphWidget)
        layout.addWidget(self.gain)
        layout.addWidget(self.gain_label)
        layout.addWidget(self.treshold)
        layout.addWidget(self.treshold_label)
        layout.addWidget(self.range)
        layout.addWidget(self.range_label)
        layout.addWidget(self.trackingWidget)
        

        self.gain.valueChanged.connect(self.gainChanged)
        self.setLayout(layout)
        # Create a central widget for the layout
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
    
        # self.slider.setFixedWidth(200)
        # self.slider.setFixedHeight(50
        # self.data_line =  self.graphWidget.plot(self.delegate.times, self.delegate.vals, pen=pen)

        # self.timer2 = QtCore.QTimer()
        # self.timer2.setInterval(10)
        # self.timer2.timeout.connect(self.readByte)
        # self.timer2.start()
        self.timer = QtCore.QTimer()
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.update_plot_data)
        self.timer.start()
        self.ser = serial.Serial('/dev/ttyUSB1', 250000, timeout=0.1)
        self.sync = False
        self.calibration = []
        #TO DEFINE
        self.cols = 11
        self.rows = 11

        #To determine the number of cluesters
        self.model = KMeans(n_clusters=5)
        self.cells = self.cols*self.rows
        self.buffer = np.zeros(self.cells)
        self.mean = 10000
        self.std = 10000

        self.startTime = time.time()
    
    def rangeChanged(self):
        range = self.range.value()
        self.range_label.setText("Range: " + str(range))
     

    def tresholdChanged(self):
        treshold = self.treshold.value()
        self.treshold_label.setText("Treshold: " + str(treshold))
      

    def gainChanged(self):
        gain = self.gain.value()
        self.gain_label.setText("Gain: " + str(gain))
        

    def mouseClicked(self, event):
        self.calibration = []
        print("Mouse clicked at: ", event.scenePos())

    def readSync(self):
        while(not self.sync):
            data = self.ser.readline()
            if len(data) < 3:
                print("No sync")
                continue
            buffer = data.split(b":")
            if len(buffer) ==5:
                self.cols = int(buffer[1])
                self.rows = int(buffer[3])
                self.cells = self.cols*self.rows
                self.sync = True
                print(self.cols, self.rows, self.cells)
                for i in range(5):
                    print(buffer[i])
                print("Synced")
                return
    
    def readString(self):
        data = self.ser.readline()
        if len(data) >= self.cells:
            data = data.decode("utf-8")
            data = data.split(",")
            try:
                data = [int(i) for i in data]
            except:
                self.sync = False
                self.readSync()
            # print(data)
            if len(data)==self.cells:
                self.buffer = np.array(data)

    def readByte(self):
        data = self.ser.read(self.cells+2)
        if len(data)==self.cells+2:
            lowbyteMin = data[1]
            highbyteMin = data[0]
            min = int(highbyteMin)<<8 + int(lowbyteMin)

            self.buffer = np.zeros(self.cells)
            for i in range(self.cells):
                self.buffer[i] = data[i+2] + min

        else:
            print("No data", data)
            # self.sync = False
            # self.readSync()
            # self.ser.write(b"s")

    def setup(self):
        print("Setup")
        # while not self.sync:
        #     print("Syncing")
        self.readSync()
        self.ser.write(str(self.gain.value()).encode())

    def update_plot_data(self):
        gain_bytes = str(self.gain.value()).encode()
        self.ser.write(gain_bytes)
        #self.ser.write(b"s")
        # self.readByte()
        self.readString()
        #  reashaep the data
        self.buffer = self.buffer.reshape(self.rows, self.cols)
        #  plot the data*
        # print(self.buffer.shape)
        temp = self.buffer
        if len(self.calibration)<80:
            self.calibration.append(self.buffer)
            self.mean = np.mean(np.array(self.calibration), axis=0)
            self.std = np.std(np.array(self.calibration), axis=0)
            if len(self.calibration)%10==0:
                print(len(self.calibration))
  
        # temp = np.clip(temp, 0, 1000)
        temp = temp - self.mean
        #print(temp)
        temp[temp<self.treshold.value()] = 0
        tracking = GravityCenter(self, temp)
        print(temp, type(temp))
        
        #disable auto levels

        image = pg.ImageItem(temp, autoLevels=False, autoRange=False, autoHistogramRange=False, levels=(0, self.range.value()))
        ImageTrack = pg.ImageItem(tracking, autoLevels=False, autoRange=False, autoHistogramRange=False, levels=(0, 1)) 
        self.trackingWidget.plotItem.clear()
        self.trackingWidget.plotItem.addItem(ImageTrack)
        self.graphWidget.plotItem.clear()
        self.graphWidget.plotItem.addItem(image)
        # self.graphWidget.setImage(self.buffer, autoLevels=False, autoRange=False, autoHistogramRange=False, levels=(0, 1000))
        

def GravityCenter(self, temp : np.ndarray):
    posx,posy = 0,0
    center = np.zeros((self.rows, self.cols))
    moyenne = np.mean(temp)  
    print("moyenne value : ", moyenne)
    if moyenne > 1 :
        for x in range(self.cols):
            for y in range(self.rows):
                #print("somme : ",temp.sum(), " type : ", type(temp.sum()))
                posx += (1/temp.sum())*temp[x,y]*x
            
                posy += (1/temp.sum())*temp[x,y]*y
        
        center[int(posx),int(posy)] = 1
    return center

def ElbowCluesters(self,temp : np.ndarray):
    #determine the number of cluesters in temp matrix with the elbow method 
    #return the number of cluesters
    elbowtracking = np.zeros((self.rows, self.cols))
    #get the index of each element not null of temp
    temp = temp.reshape(1,-1)
    temp = temp[0]
    points = np.array([[x%self.cols,x//self.rows] for x in range(len(temp)) if temp[x]!=0])
    print(points)

    visualizer = KElbowVisualizer(self.model, k=5)
    visualizer.fit(points)
    #visualizer.show()
    nbclusters = visualizer.elbow_value_
    print("Nb of cluesters : ",nbclusters)
    if nbclusters != None : 
        kmeans_model = KMeans(n_clusters=int(nbclusters)).fit(points)
        kmeans_model.fit(points)
        #find position of the center of the cluster 
        for i in kmeans_model.cluster_centers_:
            elbowtracking[int(i[0]),int(i[1])] = 1

    
    # kmeans_model.fit(points)


    return elbowtracking, nbclusters
    
def ElbowCluestersV2(self,temp : np.ndarray, ncluesters : int):
    for i in range(ncluesters):
        kmeanModel = KMeans(n_clusters=i+1).fit(temp)
        kmeanModel.fit(temp)


#method to determine number of cluesters with the silhouette method
def SilhouetteCluesters(self,temp : np.ndarray):
    #get the index of each element not null of temp
    #temp = temp.reshape(1,-1)
    #temp = temp[0]
    #temp = temp.tolist()
    #temp = np.array(temp)
    #temp = temp.reshape(self.rows, self.cols)
    #print(temp)


  

    y = [[i,v] for i,v in range(len(temp)) if temp[i] != 0]
    visualizer = SilhouetteVisualizer(self.model, colors='yellowbrick')
    visualizer.fit(temp)
    return visualizer.elbow_value_
      



#attach the signal
signal.signal(signal.SIGINT, SignalHandler)



app = QtWidgets.QApplication([])
w = MainWindow()
w.setup()
w.show()
app.exec_()
# dev.disconnect()s
while True:
    # if dev.waitForNotifications(1.0):
        # handleNotification() was called
        # continue
    # print("Waiting...")
    pass