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
        if hist is None:
            return
        x = np.arange(len(hist))
        plt.bar(x, hist, color='blue')
        plt.title(title)
        plt.show()    
    
    # Anthony
    def to_gray(self, img = None): # 圖像轉灰階 input: (3,x,x) BGR image output: (x, x) gray image        
        if img is None:
            return
        # opencv 預設排序不是RGB => BGR
        R = img[2,:,:]
        G = img[1,:,:]
        B = img[0,:,:]

        # gray = np.mean(img, axis=0).astype(np.uint8)
        # self.gray = 0.299 * R + G * 0.587 + B * 0.114
        gray = np.around(0.299 * R + G * 0.587 + B * 0.114) #opencv version
        gray = gray.astype(np.uint8) # 圖像只有正數
        
        return gray

    # Andy
    def to_histogram(self, img = None): #圖像直方圖 input: gray image; output: array[255] histogram
        if img is None:
            return
        histogram = np.array([(img == i).sum() for i in range(256)])
        return histogram
    
    # Azen
    def histogram_equalization(self, img = None): # 直方圖均衡化 input: gray; image output: gray image after histogram equalization
        if img is None:
            return
        height, width = img.shape
        histogram = self.to_histogram(img)
        histogram = self.contrast_limited(histogram, width, height)
        # self.show_histogram(title='Input image Histogram', hist = histogram)
        p = histogram / (height * width - 1) * 1.0
        # self.show_histogram(title='Input image PDF', hist = p)
        cdf = np.zeros(len(p))
        cdf[0] = p[0]
        for i in range(1, len(p)):
            cdf[i] = cdf[i - 1] + p[i]
        cdf = cdf - np.min(cdf)
        # self.show_histogram(title='Input image CDF', hist = cdf)
        cdf = np.around((256 - 1) * cdf)
        # self.show_histogram(title='T(r)', hist = cdf)
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

    # Anthony
    def contrast_limited(self, histogram = None, width_block = None, height_block = None): # input: array[255] output: array[255]
        """ 
            將直方圖限制的數值切出，並對齊取平均在加入至直方圖的每個點上
            Cut out the values ​​of the histogram limits and 
            average them over each point added to the histogram.
        """
        if histogram is None or width_block is None or height_block is None:
            return
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

    # Andy
    def clahe(self, img = None, subset_img  = (128, 128)): # input: gray image output: gray image
        if (img is None):
            return
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
    
    # Anthony
    def homomorphic_filter(self, gary_image, gamma_l=0.5, gamma_h=2.0, c=2.0, d0=20):

        # 將圖像轉換為浮點數類型
        img_float = np.float64(gary_image)

        # 對圖像進行對數轉換
        img_log = np.log(img_float + 1)

        # 傅立葉變換
        f_img = np.fft.fft2(img_log)#自己的傅立葉轉換還在試

        # 中心化傅立葉變換
        # f_img_shifted = np.fft.fftshift(f_img)

        # 創建濾波器
        rows, cols = gary_image.shape
        # rows, cols = 640,480
        H_u_v = np.zeros((rows,cols))
        for i in range(rows):
            for j in range(cols):
                D = (i - rows // 2) **2 + (j - cols//2)**2
                # D = (i) **2 + (j)**2
                H_u_v[i][j] = (gamma_h - gamma_l) * (1 - np.exp(-c * (D / (d0**2)))) + gamma_l
                
        # x, y = np.meshgrid(np.arange(-cols//2, cols//2), np.arange(-rows//2, rows//2))
        # H = (gamma_h - gamma_l) * (1 - np.exp(-c * ((x**2 + y**2) / (d0**2)))) + gamma_l
        

        # 濾波
        # filtered_image = np.fft.ifft2(np.fft.ifftshift(H_u_v * f_img_shifted))
        filtered_image = np.fft.ifft2(H_u_v * f_img )

        # 反對數轉換
        filtered_image = np.exp(np.real(filtered_image)) - 1

        # 正規化 0-255範圍
        filtered_image = (filtered_image - np.min(filtered_image)) / (np.max(filtered_image) - np.min(filtered_image)) * 255

        # 轉換為8位整數
        filtered_image = np.uint8(filtered_image)

        return filtered_image


    def apply_noise(self, gray_image, mean=100,sigma=10):
        row = gray_image.shape[0]
        col = gray_image.shape[1]
        gauss = np.random.normal(mean,sigma,(row,col))
        noisy = gray_image + gauss
        noisy = np.clip(noisy, 0, 255)
        noisy = noisy.astype(np.uint8)

        return noisy

    
    def conv(self, gray_image, filter = (1 / 9) * np.ones(shape = (3, 3))): # input: gray img, filter output: the image after convolve
        
        # Step1. 定義 gray_image、filters 的大小，並計算出 filters 在圖片上需要掃描的次數
        img_width = gray_image.shape[1]
        img_height = gray_image.shape[0]
        filter_width = filter.shape[1]
        filter_height = filter.shape[0]
        target_img_width = img_width - filter_width + 1
        target_img_height = img_height - filter_width + 1
        print(img_width, img_height, filter_width, filter_height, target_img_width, target_img_height)
        print(filter)

        # Step2. 先掃描 height，再掃描 width
        targetImg = list()
        for H in range(target_img_height):
            oneWidthImg = list()
            for W in range(target_img_width):
                targetPixel = np.sum(gray_image[H:H+filter_height, W:W+filter_width] * filter)
                oneWidthImg.append(targetPixel)

            targetImg.append(oneWidthImg)
        return np.array(targetImg).astype("uint8")