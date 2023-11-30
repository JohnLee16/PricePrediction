import pytesseract
import cv2
from PIL import Image
import numpy as np

def gamma(img):
    # 归1
    Cimg = img / 255
    # 伽玛变换
    gamma = 1.5
    O = np.power(Cimg, gamma)
    O = O * 255
    # 效果
    cv2.imshow("gamma", O)
    cv2.waitKey(0)
    
def hist(source):
    img = cv2.imread(source, cv2.IMREAD_GRAYSCALE)
    # 求出img 的最大最小值
    Maximg = np.max(img)
    Minimg = np.min(img)
    # 输出最小灰度级和最大灰度级
    Omin, Omax = 0, 255
    # 求 a, b
    a = float(Omax - Omin) / (Maximg - Minimg)
    b = Omin - a * Minimg
    # 线性变换
    O = a * img + b
    O = O.astype(np.uint8)
    cv2.imshow('enhance', O)
    #cv2.imwrite('hist.png', O, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def hist_auto(img):
    img = cv2.resize(img, None, fx=10, fy=10, interpolation=cv2.INTER_AREA)
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 限制对比度的自适应阈值均衡化
    dst = clahe.apply(img)
    # 使用全局直方图均衡化
    equa = cv2.equalizeHist(img)
    # 分别显示原图，CLAHE，HE
    #cv.imshow("img", img)
    return equa

def calcGrayHist(I):
    # 计算灰度直方图
    h, w = I.shape[:2]
    grayHist = np.zeros([256], np.uint64)
    for i in range(h):
        for j in range(w):
            grayHist[I[i][j]] += 1
    return grayHist
    
def equalHist(img):
    import math
    # 灰度图像矩阵的高、宽
    h, w = img.shape
    # 第一步：计算灰度直方图
    grayHist = calcGrayHist(img)
    # 第二步：计算累加灰度直方图
    zeroCumuMoment = np.zeros([256], np.uint32)
    for p in range(256):
        if p == 0:
            zeroCumuMoment[p] = grayHist[0]
        else:
            zeroCumuMoment[p] = zeroCumuMoment[p - 1] + grayHist[p]
    # 第三步：根据累加灰度直方图得到输入灰度级和输出灰度级之间的映射关系
    outPut_q = np.zeros([256], np.uint8)
    cofficient = 256.0 / (h * w)
    for p in range(256):
        q = cofficient * float(zeroCumuMoment[p]) - 1
        if q >= 0:
            outPut_q[p] = math.floor(q)
        else:
            outPut_q[p] = 0
    # 第四步：得到直方图均衡化后的图像
    equalHistImage = np.zeros(img.shape, np.uint8)
    for i in range(h):
        for j in range(w):
            equalHistImage[i][j] = outPut_q[img[i][j]]

    return equalHistImage
    
def line_trans_img(img,coffient):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    out = 2*img
    # 像素截断
    out[out>255] = 255
    out = np.around(out)
    return out

def get_train_number(img):
    # lower_color = np.array([35, 43, 46])
    # upper_color = np.array([77, 255, 255])
    # lower_color = np.array([0, 43, 46])
    # upper_color = np.array([42, 255, 255])
    # hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([2, 0, 0], dtype = "uint8") 
    upper_red = np.array([255, 185, 255], dtype = "uint8")
    mask = cv2.inRange(img, lower_red, upper_red)
    res_img = cv2.bitwise_and(img, img, mask=mask)
    gray_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (1, 1), 0)

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 4)
    # cv2.imshow("binary image", thresh)
    # cv2.waitKey(0)
    config = '-l eng --psm 7 -c tessedit_char_whitelist=GFDZ0123456789,*'
    text = pytesseract.image_to_string(thresh, config=config)
    # ## there are another method to process images
    # # gamma_img = gamma(gray_img)
    # equa_img = hist_auto(gray_img)
    # # cv2.imshow("equal image", equa_img)
    # # cv2.waitKey(0)


    # ret, binary = cv2.threshold(equa_img, 200, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # # cv2.imshow("binary image", binary)
    # # cv2.waitKey(0)
    # # kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]], np.float32)
    # # dst = cv2.filter2D(binary, -1, kernel=kernel)
    # # cv2.imshow("custom_blur_demo", dst)

    # # # 形态学操作   腐蚀  膨胀
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    # dilate = cv2.dilate(binary, kernel, iterations=3)

    # # _, labels, stats, centroids = cv2.connectedComponentsWithStats(dilate)
    # # i=0
    # # for istat in stats:
    # #     if istat[4]<120:
    # #         #print(i)
    # #         print(istat[0:2])
    # #         if istat[3]>istat[4]:
    # #             r=istat[3]
    # #         else:r=istat[4]
    # #         cv2.rectangle(dilate,tuple(istat[0:2]),tuple(istat[0:2]+istat[2:4]) , 0,thickness=-1)  # 26
    # #     i=i+1
    
    # # cv2.imshow('dilate', dilate)
    # # cv2.waitKey(0)
    
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
    # erode = cv2.erode(dilate, kernel, iterations=2)
    
    # # 逻辑运算  让背景为白色  字体为黑  便于识别
    # cv2.bitwise_not(erode, erode)
    # cv2.imshow('erode', erode)
    # cv2.waitKey(0)

    # test_message = Image.fromarray(gray_img)
    # config = '--psm 4 -c tessedit_char_whitelist=G0123456789,'
    # text = pytesseract.image_to_string(test_message, config=config)
    return text


def get_train_number_new(img):
    config = '-l eng --psm 7 -c tessedit_char_whitelist=GFDZTJ0123456789,*:'
    text = pytesseract.image_to_string(img, config=config)
    return text

def get_time_stamp(img):
    # img = cv2.resize(img, None, fx=10, fy=10, interpolation=cv2.INTER_AREA)
    # get gray image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # # binary
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # cv2.imwrite('./data/graytime.jpg', gray)
    # cv2.imwrite('./data/binarytime.jpg', binary)
    # cv2.imshow('dilate', binary)
    # cv2.waitKey(0)

    # # # # 形态学操作   腐蚀  膨胀
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
    # dilate = cv2.dilate(binary, kernel, iterations=1)

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
    # erode = cv2.erode(dilate, kernel, iterations=1)

    # # cv2.imshow('dilate', erode)
    # # cv2.waitKey(0)
    # # 逻辑运算  让背景为白色  字体为黑  便于识别
    # cv2.bitwise_not(erode, erode)

    # # cv2.imshow('binary-image', erode)
    # # cv2.waitKey(0)
    # 识别
    config = '-l eng --psm 7 -c tessedit_char_whitelist=0123456789,*: '
    # test_message = Image.fromarray(erode)
    text = pytesseract.image_to_string(gray, config=config)
    # print("当前识别时间是： ", text)
    return text

def transfer_string_to_int(track_string):
    result = 0
    for cc in track_string:
        if cc.isdigit():
            result = result * 10 + int(cc)
    return result
    
if __name__ == "__main__":
    image = cv2.imread("./data/test2.png")
    # text = get_train_number_new(image)
    # print(text)
    print(get_time_stamp(image))
    # cv2.imshow("origin ", image)
    # cv2.waitKey(0)

