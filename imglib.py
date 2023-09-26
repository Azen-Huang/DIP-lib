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
    
    def show_histogram(self, title = 'histogram', hist = None):
        if hasattr(self, 'histogram') and hist is None:
            hist = self.histogram
        if hist is not None:
            x = np.arange(len(hist))
            plt.bar(x, hist, color='blue')
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
        if hasattr(self, 'gray') and img is None:
            img = self.gray
        if img is not None:
            self.histogram = np.array([(img == i).sum() for i in range(256)])
            return self.histogram
    
    # Azen
    def histogram_equalization(self, img): # 直方圖均衡化 input: gray; image output: gray image after histogram equalization
        # Azen
        height, width = img.shape
        histogram = self.to_histogram(img)
        histogram = self.contrast_limited(histogram, width, height)
        # self.show_histogram(title='contrast limited', hist = histogram)
        p = histogram / (height * width - 1) * 1.0
        cdf = np.zeros(len(p))
        cdf[0] = p[0]
        for i in range(1, len(p)):
            cdf[i] = cdf[i - 1] + p[i]
        cdf = cdf - np.min(cdf)
        
        cdf = np.around((256 - 1) * cdf)
        
        # cdf = (256 - 1) * (np.cumsum(self.histogram)/(img.size * 1.0))
        # self.show_histogram('CDF', hist=cdf)
        cdf = cdf.astype('uint8')
        
        uniform_gray = np.zeros(img.shape, dtype='uint8')  # Note the type of elements
        for i in range(height):
            for j in range(width):
                uniform_gray[i,j] = cdf[img[i,j]]
        
        self.process_img = uniform_gray
        # self.show('uniform_gray', uniform_gray)
        return uniform_gray

    # anthony
    def contrast_limited(self,histogram,width_block,height_block): # input: array[255] output: array[255]
        """ 
            將直方圖限制的數值切出，並對齊取平均在加入至直方圖的每個點上
            Cut out the values ​​of the histogram limits and 
            average them over each point added to the histogram.
        """
        average = width_block * height_block // 256
        Limit = 2 * average
        # print("Limit",Limit)
        cutting = 0
        for i in range(len(histogram)):
            if histogram[i] > Limit:
                cutting += histogram[i] - Limit
                histogram[i] = Limit
            
        cutting = cutting // 256
        
        for i in range (len(histogram)):
            histogram[i] = histogram[i] + cutting

        return histogram
        
    
    # def clculate_CDF(self,array):
    #     total = np.sum(array)
    #     cdf_value = np.cumsum(array) / total #cumsum 累加 prefix sum

    #     return cdf_value

    # Andy
    def clahe(self, img, subset_img  = (512, 512)): # input: gray image output: gray image
        """
        subset_img: (height, width)
        to_histogram
        histogram_equalization
        contrast_limited
        to_gray
        """

        # Step1. Split images into four subset
        numHeight = img.shape[0] // subset_img[0]
        numWidth = img.shape[1] // subset_img[1]
        print(img.shape, numHeight, numWidth)

        images = list()
        for oneHeight in range(numHeight):
            for oneWidth in range(numWidth):
                images.append(
                    img[oneHeight*subset_img[0]:(oneHeight+1)*subset_img[0],  oneWidth*subset_img[0]:(oneWidth+1)*subset_img[0]]
                )
        
        print(len(images))
        # Step2. Each images is converted into histogram equalization
        grayImg = [
            self.histogram_equalization(img = oneImg)
            for oneImg in images
        ]

        # for img in grayImg:
        #     self.show('img', img)

        # Step3. Merge image
        i = 0
        newImg = np.zeros(shape = (grayImg[0].shape[0] * numHeight, grayImg[0].shape[0] * numWidth), dtype=np.uint8)
        for oneHeight in range(numHeight):
            for oneWidth in range(numWidth):
                newImg[oneHeight*subset_img[0]:(oneHeight+1)*subset_img[0],  oneWidth*subset_img[0]:(oneWidth+1)*subset_img[0]] = grayImg[i]
                i += 1
        
        self.show('img', newImg)
        return newImg