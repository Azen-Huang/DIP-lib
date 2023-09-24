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

    def show_histogram(self, title = 'histogram'):
        if hasattr(self, 'histogram'):
            x = np.arange(len(self.histogram))
            plt.bar(x, self.histogram, color='blue')
            plt.title(title)
            plt.show()
    
    # Anthony
    def to_gray(self): # 圖像轉灰階 input: (3,x,x) BGR image output: (x, x) gray image        
        # opencv 預設排序不是RGB => BGR
        R = self.process_img[2,:,:]
        G = self.process_img[1,:,:]
        B = self.process_img[0,:,:]

        # self.gray = np.mean(self.process_img,axis=0).astype(np.uint8)
        # self.gray = 0.299 * R + G * 0.587 + B * 0.114
        self.gray = np.around(0.299 * R + G * 0.587 + B * 0.114) #opencv version
        self.gray = self.gray.astype(np.uint8) # 圖像只有正數
        
        return self.gray

    # 建安
    def to_histogram(self, img = None): #圖像直方圖 input: gray image; output: array[255] histogram
        # 建安
        if hasattr(self, 'gray') and img is None:
            img = self.gray
        if img is not None:
            self.histogram = np.array([(img == i).sum() for i in range(256)])
            return self.histogram
    
    # Azen
    def histogram_equalization(self): # 直方圖均衡化 input: gray; image output: gray image after histogram equalization
        # Azen
        height, width = self.gray.shape
        p = self.histogram / (height * width - 1) * 1.0
        cdf = np.zeros(len(p))
        cdf[0] = p[0]
        for i in range(1, len(p)):
            cdf[i] = cdf[i - 1] + p[i]
        cdf = cdf - np.min(cdf)
        cdf = np.around((256 - 1) * cdf)
        
        # cdf = (256 - 1) * (np.cumsum(self.histogram)/(self.gray.size * 1.0))

        cdf = cdf.astype('uint8')
        self.histogram = cdf
        self.show_histogram('CDF')
        uniform_gray = np.zeros(self.gray.shape, dtype='uint8')  # Note the type of elements
        for i in range(height):
            for j in range(width):
                uniform_gray[i,j] = cdf[self.gray[i,j]]
        
        self.process_img = uniform_gray
        return uniform_gray
