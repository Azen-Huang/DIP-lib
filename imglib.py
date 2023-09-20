import numpy as np
import matplotlib.pyplot as plt
import cv2

class imglib:
    def __init__(self, img) -> None:
        self.original_img = img
        self.process_img = np.transpose(self.original_img, (2, 0, 1))
        self.row_size = len(img[0])
        self.col_size = len(img)
    
    def set_process_img(self, img):
        self.process_img = img

    def show(self, title, img):
        screen_width = 1500
        screen_height = 800
        x = (screen_width - img.shape[0]) // 2
        y = (screen_height - img.shape[1]) // 2
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.moveWindow(title, x, y)
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def info(self):
        print('-' * 20 + 'Info' + '-' * 20)
        print('Original image shape: ', self.original_img.shape)
        print('Process image shape: ', self.process_img.shape)
        print('gray image shape: ', self.gray.shape) if hasattr(self,'gray') else None
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
        self.gray = np.around(0.299 * R + G * 0.587 + B * 0.114) #opencv version
        self.gray = self.gray.astype(np.uint8) # 圖像只有正數
        
        return self.gray

    def to_histogram(self): #圖像直方圖 input: 2D array; output: array[255]
        if hasattr(self, 'gray'):
            self.histogram = np.array([(self.gray == i).sum() for i in range(256)])
            return self.histogram

    def histogram_equalization(self): 
        pass
