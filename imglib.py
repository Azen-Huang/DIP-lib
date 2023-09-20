import numpy as np
import matplotlib.pyplot as plt
class imglib:
    def __init__(self, img) -> None:
        self.original_img = img
        self.process_img = np.transpose(self.original_img, (2, 0, 1))
        self.row_size = len(img[0])
        self.col_size = len(img)
    
    def set_process_img(self, img):
        self.process_img = img
        
    def info(self):
        print('-' * 20 + 'Info' + '-' * 20)
        print('Original image shape: ', self.original_img.shape)
        print('Process image shape: ', self.process_img.shape)
        print('-' * 44)

    def show_histogram(self):
        if hasattr(self, 'histogram'):
            x = np.arange(len(self.histogram))
            plt.bar(x, self.histogram, color='blue')
            plt.title('histogram')
            plt.show()
    
    def to_gray(self): #圖像轉灰階 input: (3,x,x) BGR image output: (x, x) gray image
        #opencv 預設排序不是RGB => BGR
        R = self.process_img[2,:,:]
        G = self.process_img[1,:,:]
        B = self.process_img[0,:,:]
        # self.Gray = np.mean(self.process_img,axis=0).astype(np.uint8)
        # self.Gray = 0.299 * R + G * 0.587 + B * 0.114
        self.gray = np.around(0.299 * R + G * 0.587 + B * 0.114)
        self.gray = self.gray.astype(np.uint8) # 圖像只有正數
        
        return self.gray

    def to_histogram(self): #圖像直方圖 input: 2D array; output: array[255]
        if hasattr(self, 'gray'):
            self.histogram = np.array([(self.gray == i).sum() for i in range(256)])
            return self.histogram

    def histogram_equalization(self): 
        pass
