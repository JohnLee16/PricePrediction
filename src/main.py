import cv2
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from exportexcel import WriteExcel
from tqdm import *
from ocrmodule import get_time_stamp, get_train_number_new, transfer_string_to_int
import time
import re

bg_color = [197, 102, 6]
threshold = 3000

waiting_in_light1 = False
waiting_in_light2 = False
waiting_in_light3 = False
waiting_out_light1 = False
waiting_out_light2 = False
waiting_out_light3 = False

train_in_light1 = False
train_in_light2 = False

time_region = [[90, 109], [106, 169]]

def calc_diff(pixel):
    '''
    计算pixel与背景的离差平方和, 作为当前像素点与背景相似程度的度量
    '''
    return (pixel[0]-bg_color[0])**2 + (pixel[1]-bg_color[1])**2 + (pixel[2]-bg_color[2])**2

def remove_bg(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA) #将图像转成带透明通道的BGRA格式
    h, w = image.shape[0:2]
    for i in range(h):
        for j in range(w):
            if calc_diff(image[i][j]) < threshold:
                #若果logo[i][j]为背景，将其颜色设为白色，且完全透明
                image[i][j][0] = 255
                image[i][j][1] = 255
                image[i][j][2] = 255
                image[i][j][3] = 0
 
    cv2.imshow("background", image)

def get_time1(last_frame, curr_frame, time_frame, track_numbers):
    # time1_region = [[402, 1120],[622, 1996],[406, 321],[603, 846], [298, 103],[352, 1103],]
    # 3条进入的轨道
    track_region_dict = {
        1: [[299, 852], [344, 1207]],
        2: [[354, 855], [385, 1205]],
        3: [[264, 890], [295, 1204]],
        4: [[408, 856], [431, 1221]],
        5: [[223, 937], [250, 1167]],
        6: [[454, 872], [476, 1182]],
        7: [[174, 986], [206, 1168]],
        8: [[502, 914], [525, 1146]],
        9: [[127, 995], [159, 1164]],
        10: [[540, 935], [576, 1121]],
        12: [[585, 933], [624, 1122]]
    }
    # time1_region = [[225, 114],[352, 890],[400, 1066],[568, 1721],[340, 1306],[405, 1863]]
    
    diff = cv2.absdiff(last_frame, curr_frame)
    lower_white_color = np.array([200, 200, 200], dtype = "uint8")
    upper_white_color = np.array([255, 255, 255], dtype = "uint8")

    mask = cv2.inRange(diff, lower_white_color, upper_white_color)
    diff_images = []
    time1_list = []
    if mask.max() > 0:
        # for key in track_numbers:
        diff_images.append(mask[track_region_dict[track_numbers][0][0]: track_region_dict[track_numbers][1][0], track_region_dict[track_numbers][0][1]: track_region_dict[track_numbers][1][1]])
        # cv2.imshow('d', mask[track_region_dict[track_numbers][0][0]: track_region_dict[track_numbers][1][0], track_region_dict[track_numbers][0][1]: track_region_dict[track_numbers][1][1]])
        # cv2.waitKey(0)
        edges = cv2.Canny(diff_images[len(diff_images) - 1], 50, 150, apertureSize=3)  # apertureSize是sobel算子大小，只能为1,3,5，7
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=100,maxLineGap=10)  #函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
        if lines is not None:
            return get_time_stamp(time_frame[time_region[0][0]: time_region[1][0], time_region[0][1]:time_region[1][1]]), track_numbers
            # time1_list.append(get_time_stamp(time_frame[time_region[0][0]: time_region[1][0], time_region[0][1]:time_region[1][1]]), track_numbers)

        # for region_index in range(0, len(time1_region), 2):
        #     diff_images.append(mask[time1_region[region_index][0]: time1_region[region_index + 1][0], time1_region[region_index][1]: time1_region[region_index + 1][1]])
        #     # gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        #     edges = cv2.Canny(diff_images[len(diff_images) - 1], 50, 150, apertureSize=3)  # apertureSize是sobel算子大小，只能为1,3,5，7
        #     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=100,maxLineGap=10)  #函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
        #     if lines is not None:
        #         time1_list.append(get_time_stamp(time_frame[time_region[0][0]: time_region[1][0], time_region[0][1]:time_region[1][1]]))
    return None, track_numbers

def get_time1new(last_frame, curr_frame, time_frame, track_number):
    REGION_PIEXL_NR = 15
    train_03 = curr_frame[201:234, 525:650]
    train_13 = curr_frame[201:234, 655:780]
    train_06 = curr_frame[138: 171, 2055: 2180]
    train_16 = curr_frame[138: 171, 2186: 2301]
    train_21 = curr_frame[512: 545, 180: 305]
    train_31 = curr_frame[512: 545, 310: 435]
    train_24 = curr_frame[540: 573, 1425: 1550]
    train_34 = curr_frame[540: 573, 1555: 1680]
    
    

    train_number_string_list = []
    lower_green = np.array([48,43,46])
    upper_green = np.array([65,255,255])
    lower_yellow = np.array([26,43,46])
    upper_yellow = np.array([34,255,255])
    train03_hsv_image = cv2.cvtColor(train_03, cv2.COLOR_BGR2HSV)
    if cv2.countNonZero(cv2.inRange(train03_hsv_image, lower_green, upper_green)) > REGION_PIEXL_NR:
        train_number_string_list.append(get_train_number_new(train_03))

    # train_number_string_list.append(get_train_number_new(train_11))
    # train_number_string_list.append(get_train_number_new(train_12))
    train13_hsv_image = cv2.cvtColor(train_13, cv2.COLOR_BGR2HSV)
    if cv2.countNonZero(cv2.inRange(train13_hsv_image, lower_green, upper_green)) > REGION_PIEXL_NR:
        train_number_string_list.append(get_train_number_new(train_13))

    train06_hsv_image = cv2.cvtColor(train_06, cv2.COLOR_BGR2HSV)
    if cv2.countNonZero(cv2.inRange(train06_hsv_image, lower_green, upper_green)) > REGION_PIEXL_NR:
        train_number_string_list.append(get_train_number_new(train_06))

    train16_hsv_image = cv2.cvtColor(train_16, cv2.COLOR_BGR2HSV)
    if cv2.countNonZero(cv2.inRange(train16_hsv_image, lower_green, upper_green)) > REGION_PIEXL_NR:
        train_number_string_list.append(get_train_number_new(train_16))

    train21_hsv_image = cv2.cvtColor(train_21, cv2.COLOR_BGR2HSV)
    if cv2.countNonZero(cv2.inRange(train21_hsv_image, lower_green, upper_green)) > REGION_PIEXL_NR:
        train_number_string_list.append(get_train_number_new(train_21))
    
    train31_hsv_image = cv2.cvtColor(train_31, cv2.COLOR_BGR2HSV)
    if cv2.countNonZero(cv2.inRange(train31_hsv_image, lower_green, upper_green)) > REGION_PIEXL_NR:
        train_number_string_list.append(get_train_number_new(train_31))
    
    train24_hsv_image = cv2.cvtColor(train_24, cv2.COLOR_BGR2HSV)
    if cv2.countNonZero(cv2.inRange(train24_hsv_image, lower_green, upper_green)) > REGION_PIEXL_NR:
        train_number_string_list.append(get_train_number_new(train_24))
    
    train34_hsv_image = cv2.cvtColor(train_34, cv2.COLOR_BGR2HSV)
    if cv2.countNonZero(cv2.inRange(train34_hsv_image, lower_green, upper_green)) > REGION_PIEXL_NR:
        train_number_string_list.append(get_train_number_new(train_34))
    

def get_time2(last_frame, curr_frame, time_frame):
    in_port1 = curr_frame[305:320, 378:410]
    last_in_port1 = last_frame[305:320, 378:410]
    in_port2 = curr_frame[383:399, 1784:1815]
    last_in_port2 = last_frame[383:399, 1784:1815]
    in_port3 = curr_frame[428:443, 1649:1677]
    last_in_port3 = last_frame[428:443, 1649:1677]


    lower_red = np.array([2, 0, 100], dtype = "uint8")
    upper_red = np.array([10, 0, 255], dtype = "uint8")
    lower_green = np.array([0, 100, 0], dtype=np.uint8)
    upper_green = np.array([100, 255, 100], dtype=np.uint8)
    # lower_yellow = np.array([30, 180, 180], dtype = "uint8") 
    # upper_yellow = np.array([80, 255, 255], dtype = "uint8")
    lower_yellow = np.array([0, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([100, 255, 255], dtype=np.uint8)
    global waiting_in_light1
    global waiting_in_light2
    global waiting_in_light3

    if cv2.inRange(in_port1, lower_yellow, upper_yellow).max() > 20 or np.sum(cv2.inRange(in_port1, lower_green, upper_green)) > 10:
        waiting_in_light1 = False
    if not waiting_in_light1 and cv2.inRange(in_port1, lower_red, upper_red).max() > 20:
        waiting_in_light1 = True

    if cv2.inRange(in_port2, lower_yellow, upper_yellow).max() > 20:
        waiting_in_light2 = False
    if not waiting_in_light2 and cv2.inRange(in_port2, lower_red, upper_red).max() > 20:
        waiting_in_light2 = True

    if cv2.inRange(in_port3, lower_yellow, upper_yellow).max() > 20:
        waiting_in_light3 = False
    if not waiting_in_light3 and cv2.inRange(in_port3, lower_red, upper_red).max() > 20:
        waiting_in_light3 = True

    return get_time_stamp(time_frame[time_region[0][0]: time_region[1][0], time_region[0][1]:time_region[1][1]]) if waiting_in_light1 or waiting_in_light2 or waiting_in_light3  else None

def get_time3(last_frame, curr_frame, time_frame):
    # 8000 多帧
    diff = cv2.absdiff(last_frame, curr_frame)
    lower_green_color = np.array([80, 30, 80], dtype = "uint8")
    upper_green_color = np.array([130, 255, 130], dtype = "uint8")

    mask = cv2.inRange(diff, lower_green_color, upper_green_color)
    res = cv2.bitwise_and(last_frame, last_frame, mask=mask)

    average = res.mean(axis=0).mean(axis=0)
    # print("average 0, 1, 2: ", average)
    if average[1] > 0.02:
        edges = cv2.Canny(mask, 50, 150, apertureSize=3)  # apertureSize是sobel算子大小，只能为1,3,5，7
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=50,maxLineGap=10)  #函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
        if lines is not None:
            return get_time_stamp(time_frame[time_region[0][0]: time_region[1][0], time_region[0][1]:time_region[1][1]])

    return None

def get_time4(last_frame, curr_frame, time_frame):
    # 4条出去的轨道
    time4_region = [[348, 80],[400, 722],[404, 334],[626, 950],[242, 1272],[300, 1785],[300, 1272],[343, 1875]]
    diff = cv2.absdiff(last_frame, curr_frame)
    lower_white_color = np.array([200, 200, 200], dtype = "uint8")
    upper_white_color = np.array([255, 255, 255], dtype = "uint8")

    mask = cv2.inRange(diff, lower_white_color, upper_white_color)
    diff_images = []
    if mask.max() > 0:
        for region_index in range(0, len(time4_region), 2):
            diff_images.append(mask[time4_region[region_index][0]: time4_region[region_index + 1][0], time4_region[region_index][1]: time4_region[region_index + 1][1]])
            # gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(diff_images[len(diff_images) - 1], 50, 150, apertureSize=3)  # apertureSize是sobel算子大小，只能为1,3,5，7
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=100,maxLineGap=10)  #函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
            if lines is not None:
                return get_time_stamp(time_frame[time_region[0][0]: time_region[1][0], time_region[0][1]:time_region[1][1]])
    return None

def get_time5(last_frame, curr_frame, time_frame, track_number):
    out_port1 = curr_frame[385:397, 382:405]
    out_port2 = curr_frame[289:301, 1785:1809]
    track_leave_region = {
        1: [[333, 794], [347, 824], [304, 1242], [320, 1270]],
        2: [[382, 795], [398, 829], [354, 1239], [371, 1269]],
        3: [[287, 928], [302, 954], [258, 1194], [274, 1224]],
        4: [[427, 928], [444, 955], [399, 1152], [416, 1181]],
        5: [[243, 934], [259, 963], [215, 1196], [231, 1226]],
        6: [[472, 927], [489, 955], [444, 1149], [461, 1178]],
        7: [[197, 941], [213, 969], [171, 1194], [187, 1224]],
        8: [[520, 928], [536, 957], [492, 1103], [508, 1133]],
        9: [[151, 942], [168, 971], [123, 1193], [140, 1224]],
        10: [[564, 883], [579, 913], [536, 1149], [553, 1181]],
        12: [[610, 883], [627, 914], [581, 599], [1145, 1176]]
    }

    port1 = curr_frame[track_leave_region[track_number][0][0]:track_leave_region[track_number][1][0], track_leave_region[track_number][0][1]:track_leave_region[track_number][1][1]]
    port2 = curr_frame[track_leave_region[track_number][2][0]:track_leave_region[track_number][3][0], track_leave_region[track_number][2][1]:track_leave_region[track_number][3][1]]

    # image channel in bgr value
    lower_red = np.array([2, 0, 100], dtype = "uint8") 
    upper_red = np.array([10, 0, 255], dtype = "uint8")
    lower_green = np.array([30, 180, 180], dtype = "uint8") 
    upper_green = np.array([80, 255, 255], dtype = "uint8")

    # # inport1_mask = cv2.inRange(in_port1, lower_red, upper_red)
    # # last_inport1_mask = cv2.inRange(last_in_port1, lower_yellow, upper_yellow)
    # # inport2_mask = cv2.inRange(in_port2, lower_red, upper_red)
    # # last_inport2_mask = cv2.inRange(last_in_port2, lower_yellow, upper_yellow)
    # global waiting_out_light1
    # global waiting_out_light2
    # global waiting_out_light3

    # if not waiting_out_light1 and cv2.inRange(out_port1, lower_green, upper_green).max() > 0:
    #     waiting_out_light1 = True
 
    # if waiting_out_light1 and cv2.inRange(out_port1, lower_red, upper_red).max() > 0:
    #     waiting_out_light1 = False

    # if not waiting_out_light2 and cv2.inRange(out_port2, lower_green, upper_green).max() > 0:
    #     waiting_out_light2 = True
    # if waiting_out_light2 and cv2.inRange(out_port2, lower_red, upper_red).max() > 0:
    #     waiting_out_light2 = False

    # if not waiting_out_light3 and cv2.inRange(out_port3, lower_green, upper_green).max() > 0:
    #     waiting_out_light3 = True
    # if waiting_out_light3 and cv2.inRange(out_port3, lower_red, upper_red).max() > 0:
    #     waiting_out_light3 = False

    if cv2.inRange(port1, lower_green, upper_green).max() > 100 or cv2.inRange(port2, lower_green, upper_green).max() > 100:
        return get_time_stamp(time_frame[time_region[0][0]: time_region[1][0], time_region[0][1]:time_region[1][1]])
    return None
    
def get_time6(last_frame, curr_frame, time_frame):
    train_leave_signal_region = [[357, 311], [398, 352],
                                 [306, 1827], [347, 1868],
                                 [260, 1821], [301, 1862],
                                 [445, 19], [486, 60]]
    train_leave_signals = []
    last_train_leave_signals = []
    for region_index in range(0, len(train_leave_signal_region), 2):
        train_leave_signals.append(curr_frame[train_leave_signal_region[region_index][0]: train_leave_signal_region[region_index + 1][0], train_leave_signal_region[region_index][1]: train_leave_signal_region[region_index + 1][1]])
        last_train_leave_signals.append(last_frame[train_leave_signal_region[region_index][0]: train_leave_signal_region[region_index + 1][0], train_leave_signal_region[region_index][1]: train_leave_signal_region[region_index + 1][1]])
    

    lower_yellow_color = np.array([0, 230, 230], dtype = "uint8") 
    upper_yellow_color = np.array([60, 255, 255], dtype = "uint8")

    lower_red_color = np.array([0, 0, 200], dtype = "uint8") 
    upper_red_color = np.array([10, 10, 255], dtype = "uint8")

    
    # i = 1
    # for train_leave_signal in train_leave_signals:
    #     i += 1  

    i = 0
    for i in range(len(train_leave_signals)):
        yellow_mask = cv2.inRange(train_leave_signals[i], lower_yellow_color, upper_yellow_color)
        red_mask = cv2.inRange(train_leave_signals[i], lower_red_color, upper_red_color)
        is_leave = yellow_mask.max() > 200 and red_mask.max() > 200

        last_yellow_mask = cv2.inRange(last_train_leave_signals[i], lower_yellow_color, upper_yellow_color)
        last_red_mask = cv2.inRange(last_train_leave_signals[i], lower_red_color, upper_red_color)
        last_not_leave = (not last_yellow_mask.max() > 200) or (not last_red_mask.max() > 200)

        if last_not_leave and is_leave:
            return get_time_stamp(time_frame[time_region[0][0]: time_region[1][0], time_region[0][1]:time_region[1][1]])
    return None

def update_time1_information(time1, track_number, output):
    # for t in time1_list:
    # row_number = output[output.TrainNumber==train_info[0]].index.tolist()[0]
    row_number = output[output.TrackNumber == track_number].index.tolist()[0]
    if output.Time1.loc[row_number] == None:
        output.loc[row_number,'Time1'] = time1
        output.loc[row_number,'Status'] = 2
    return output

def update_time2_information(time2, output):
    time2_empty_row_number = output[output[['Time2']].isnull().T.any()].index.tolist()[0]
    output.loc[time2_empty_row_number,'Time2'] = time2
    output.loc[time2_empty_row_number,'Status'] = 6
    return output

def update_time3_information(time3, output):
    time3_empty_row_number = output[output[['Time3']].isnull().T.any()].index.tolist()[0]
    output.loc[time3_empty_row_number,'Time3'] = time3
    output.loc[time3_empty_row_number,'Status'] = 4
    return output

def update_time4_information(time4, output):
    time4_empty_row_number = output[output[['Time4']].isnull().T.any()].index.tolist()[0]
    output.loc[time4_empty_row_number,'Time4'] = time4
    output.loc[time4_empty_row_number,'Status'] = 5
    return output

def update_time5_information(time5, output):
    time5_empty_row_number = output[output[['Time5']].isnull().T.any()].index.tolist()[0]
    output.loc[time5_empty_row_number,'Time5'] = time5
    output.loc[time5_empty_row_number,'Status'] = 6
    return output

def update_time6_information(time6, output, train_number):
    time6_empty_row_number = output[output.TrainNumber == train_number].index.tolist()[0]
    output.loc[time6_empty_row_number,'Time6'] = time6
    output.loc[time6_empty_row_number,'Status'] = 0
    return output

def get_train_track_information(last_frame, curr_frame, time_frame, output):
    split_train_information_list = get_train_number_region(last_frame, curr_frame)
    for train_info in split_train_information_list:
        if len(train_info[0]) < 3:
            continue 
        if train_info[2] == True and train_info[0] not in output.values:
            new_data = {'TrainNumber': train_info[0],
                        'Time1': None,
                        'Time2': None,
                        'Time3': None,
                        'Time4': None,
                        'Time5': None,
                        'Time6': None,
                        'TrackNumber': train_info[1],
                        'Status': 2}

            output = output.append(new_data, ignore_index=True)
        # elif train_info[2] == True and train_info[0] in output.values:
        #     row_number = output[output.TrainNumber==train_info[0]].index.tolist()[0]
        #     if output.Time1.loc[row_number] == None:
        #         output.loc[row_number,'Time1'] = time1_string[0]
    return output
    
def get_train_number_region(last_frame, curr_frame):
    REGION_PIEXL_NR = 15
    # train_01 = curr_frame[135:168, 525:650]
    # train_02 = curr_frame[168:201, 525:650]
    train_03_curr = curr_frame[201:234, 525:650]
    train_03_last = last_frame[201:234, 525:650]
    # train_11 = curr_frame[135:168, 655:780]
    # train_12 = curr_frame[168:201, 655:780]
    train_13_curr = curr_frame[201:234, 655:780]
    train_13_last = last_frame[201:234, 655:780]
    # train_info_region1_last = last_frame[135:234, 525:780]
    # train_info_region1_curr = curr_frame[135:234, 525:780]
    
    # train_04 = curr_frame[72: 105, 2055: 2180]
    # train_05 = curr_frame[105: 138, 2055: 2180]
    train_06_curr = curr_frame[138: 171, 2055: 2180]
    train_06_last = last_frame[138: 171, 2055: 2180]
    # train_14 = curr_frame[72: 105, 2186: 2301]
    # train_15 = curr_frame[105: 138, 2186: 2301]
    train_16_curr = curr_frame[138: 171, 2186: 2301]
    train_16_last = last_frame[138: 171, 2186: 2301]
    # train_info_region2_last = last_frame[72:171, 2055:2301]
    # train_info_region2_curr = curr_frame[72:171, 2055:2301]
    
    train_21_curr = curr_frame[512: 545, 180: 305]
    train_21_last = last_frame[512: 545, 180: 305]
    # train_22 = curr_frame[545: 578, 180: 305]
    # train_23 = curr_frame[578: 611, 180: 305]
    train_31_curr = curr_frame[512: 545, 310: 435]
    train_31_last = last_frame[512: 545, 310: 435]
    # train_32 = curr_frame[545: 578, 310: 435]
    # train_33 = curr_frame[578: 611, 310: 435]
    # train_info_region3_last = last_frame[512:611, 180:435]
    # train_info_region3_curr = curr_frame[512:611, 180:435]
    
    train_24_curr = curr_frame[540: 573, 1425: 1550]
    train_24_last = last_frame[540: 573, 1425: 1550]
    # train_25 = curr_frame[573: 606, 1425: 1550]
    # train_26 = curr_frame[606: 639, 1425: 1550]
    train_34_curr = curr_frame[540: 573, 1555: 1680]
    train_34_last = last_frame[540: 573, 1555: 1680]
    # train_35 = curr_frame[573: 606, 1555: 1680]
    # train_36 = curr_frame[606: 639, 1555: 1680]
    # train_info_region4_last = last_frame[540:639, 1425:1680]
    # train_info_region4_curr = curr_frame[540:639, 1425:1680]
    

    train_number_string_list = []
    # diff = cv2.absdiff(train_info_region1_last, train_info_region1_curr)
    # if diff.max() > 0:
    # train_number_string_list.append(get_train_number_new(train_01))
    # train_number_string_list.append(get_train_number_new(train_02))
    lower_green = np.array([48,43,46])
    upper_green = np.array([65,255,255])
    lower_yellow = np.array([26,43,46])
    upper_yellow = np.array([34,255,255])

    train03_hsv_image = cv2.cvtColor(train_03_curr, cv2.COLOR_BGR2HSV)
    train03_mask = cv2.inRange(train03_hsv_image, lower_green, upper_green)
    if cv2.countNonZero(cv2.inRange(train03_hsv_image, lower_green, upper_green)) > REGION_PIEXL_NR:
        cv2.imshow("train03", train_03_curr)
        cv2.waitKey(0)
        train_number_string_list.append(get_train_number_new(train_03_curr))

    # train_number_string_list.append(get_train_number_new(train_11))
    # train_number_string_list.append(get_train_number_new(train_12))
    train13_hsv_image = cv2.cvtColor(train_13_curr, cv2.COLOR_BGR2HSV)
    if cv2.countNonZero(cv2.inRange(train13_hsv_image, lower_green, upper_green)) > REGION_PIEXL_NR:
        cv2.imshow("train13", train_13_curr)
        cv2.waitKey(0)
        train_number_string_list.append(get_train_number_new(train_13_curr))

    # diff = cv2.absdiff(train_info_region2_last, train_info_region2_curr)
    # if diff.max() > 0:
    # train_number_string_list.append(get_train_number_new(train_04))
    # train_number_string_list.append(get_train_number_new(train_05))
    train06_hsv_image = cv2.cvtColor(train_06_curr, cv2.COLOR_BGR2HSV)
    train06_mask = cv2.inRange(train06_hsv_image, lower_green, upper_green)
    if cv2.countNonZero(cv2.inRange(train06_hsv_image, lower_green, upper_green)) > REGION_PIEXL_NR:
        cv2.imshow("train06", train_06_curr)
        cv2.waitKey(0)
        train_number_string_list.append(get_train_number_new(train_06_curr))
    # train_number_string_list.append(get_train_number_new(train_14))
    # train_number_string_list.append(get_train_number_new(train_15))
    train16_hsv_image = cv2.cvtColor(train_16_curr, cv2.COLOR_BGR2HSV)
    if cv2.countNonZero(cv2.inRange(train16_hsv_image, lower_green, upper_green)) > REGION_PIEXL_NR:
        cv2.imshow("train16", train_16_curr)
        cv2.waitKey(0)
        train_number_string_list.append(get_train_number_new(train_16_curr))

    # diff = cv2.absdiff(train_info_region3_last, train_info_region3_curr)
    # if diff.max() > 0:
    train21_hsv_image = cv2.cvtColor(train_21_curr, cv2.COLOR_BGR2HSV)
    if cv2.countNonZero(cv2.inRange(train21_hsv_image, lower_green, upper_green)) > REGION_PIEXL_NR:
        cv2.imshow("train21", train_21_curr)
        cv2.waitKey(0)
        train_number_string_list.append(get_train_number_new(train_21_curr))
    # train_number_string_list.append(get_train_number_new(train_22))
    # train_number_string_list.append(get_train_number_new(train_23))
    train31_hsv_image = cv2.cvtColor(train_31_curr, cv2.COLOR_BGR2HSV)
    if cv2.countNonZero(cv2.inRange(train31_hsv_image, lower_green, upper_green)) > REGION_PIEXL_NR:
        cv2.imshow("train31", train_31_curr)
        cv2.waitKey(0)
        train_number_string_list.append(get_train_number_new(train_31_curr))
    # train_number_string_list.append(get_train_number_new(train_32))
    # train_number_string_list.append(get_train_number_new(train_33))

    # diff = cv2.absdiff(train_info_region4_last, train_info_region4_curr)
    # if diff.max() > 0:
    train24_hsv_image = cv2.cvtColor(train_24_curr, cv2.COLOR_BGR2HSV)
    if cv2.countNonZero(cv2.inRange(train24_hsv_image, lower_green, upper_green)) > REGION_PIEXL_NR:
        cv2.imshow("train24", train_24_curr)
        cv2.waitKey(0)
        train_number_string_list.append(get_train_number_new(train_24_curr))
    # train_number_string_list.append(get_train_number_new(train_25))
    # train_number_string_list.append(get_train_number_new(train_26))
    train34_hsv_image = cv2.cvtColor(train_34_curr, cv2.COLOR_BGR2HSV)
    if cv2.countNonZero(cv2.inRange(train34_hsv_image, lower_green, upper_green)) > REGION_PIEXL_NR:
        cv2.imshow("train34", train_34_curr)
        cv2.waitKey(0)
        train_number_string_list.append(get_train_number_new(train_34_curr))
    # train_number_string_list.append(get_train_number_new(train_35))
    # train_number_string_list.append(get_train_number_new(train_36))

    split_train_information_list = []
    for train_string in train_number_string_list:
        if not train_string[0].isalpha():
            continue
        split_train_information_list.append(split_train_number_track_number(train_string))

    return split_train_information_list

def has_train_coming(curr_frame):
    train_info_region1_curr = curr_frame[135:234, 525:780]
    train_info_region2_curr = curr_frame[72:171, 2055:2301]
    train_info_region3_curr = curr_frame[512:611, 180:435]
    train_info_region4_curr = curr_frame[540:639, 1425:1680]
    
    region1_has_train = train_info_region1_curr.max() > 20
    region2_has_train = train_info_region2_curr.max() > 20
    region3_has_train = train_info_region3_curr.max() > 20
    region4_has_train = train_info_region4_curr.max() > 20

    if region1_has_train or region2_has_train or region3_has_train or region4_has_train:
        return True
    else:
        return False

def has_new_train_comming(last_frame, curr_frame):
    train_info_region1_last = last_frame[135:234, 525:780]
    train_info_region2_last = last_frame[72:171, 2055:2301]
    train_info_region3_last = last_frame[512:611, 180:435]
    train_info_region4_last = last_frame[540:639, 1425:1680]

    train_info_region1_curr = curr_frame[135:234, 525:780]
    train_info_region2_curr = curr_frame[72:171, 2055:2301]
    train_info_region3_curr = curr_frame[512:611, 180:435]
    train_info_region4_curr = curr_frame[540:639, 1425:1680]
    # cv2.imshow('region1', train_info_region1_curr)
    # cv2.imshow('region2', train_info_region1_curr)
    # cv2.imshow('region3', train_info_region1_curr)
    # cv2.imshow('region4', train_info_region1_curr)
    # cv2.waitKey(0)
    diff_1 = cv2.absdiff(train_info_region1_last, train_info_region1_curr)
    diff_2 = cv2.absdiff(train_info_region2_last, train_info_region2_curr)
    diff_3 = cv2.absdiff(train_info_region3_last, train_info_region3_curr)
    diff_4 = cv2.absdiff(train_info_region4_last, train_info_region4_curr)

    region1_has_train = diff_1.max() > 100
    region2_has_train = diff_2.max() > 100
    region3_has_train = diff_3.max() > 100
    region4_has_train = diff_4.max() > 100
    if region1_has_train or region2_has_train or region3_has_train or region4_has_train:
        return True
    return False

def split_train_number_track_number(train_string):
    '''
    split the train string after ocr
    return train number, track number and if the train was selected
    '''
    train_number = ""
    track_number = ""
    in_train_number = True
    selected_train = False
    is_first_char = True
    is_second_char = False
    for cc in train_string:
        if cc == '\n':
            break
        if is_first_char:
            train_number += cc
            is_first_char = False
            is_second_char = True
        elif is_second_char:
            train_number += cc
            is_second_char = False
        elif cc == '*':
            selected_train = True
            in_train_number = False
            continue
        elif cc.isalpha():
            in_train_number = False
            track_number += cc
        elif in_train_number:
            train_number += cc
        else:
            track_number += cc
        
    return train_number, track_number, selected_train

def has_time1_empty(output):
    status1_list = output[output.Status==1].index.tolist()
    time1_empty_row_number = status1_list[0] if len(status1_list) != 0 else None
    if time1_empty_row_number is not None:
        time1_row = output.loc[time1_empty_row_number]
        return True, transfer_string_to_int(time1_row['TrackNumber'])
    else:
        return False, None

def has_time2_empty(output):
    # time2_list = output['Time2'].tolist()
    # if None in time2_list:
    #     return True
    # else:
    #     return False
    status2_list = output[output.Status==2].index.tolist()
    time2_empty_row_number = status2_list[0] if len(status2_list) != 0 else None
    if time2_empty_row_number is not None:
        time2_row = output.loc[time2_empty_row_number]
        return True, transfer_string_to_int(time2_row['TrackNumber'])
    else:
        return False, None

def has_time3_empty(output):
    status3_list = output[output.Status==3].index.tolist()
    time3_empty_row_number = status3_list[0] if len(status3_list) != 0 else None
    if time3_empty_row_number is not None:
        time3_row = output.loc[time3_empty_row_number]
        return True, transfer_string_to_int(time3_row['TrackNumber'])
    else:
        return False, None

def has_time4_empty(output):
    status4_list = output[output.Status==4].index.tolist()
    time4_empty_row_number = status4_list[0] if len(status4_list) != 0 else None
    if time4_empty_row_number is not None:
        time4_row = output.loc[time4_empty_row_number]
        return True, transfer_string_to_int(time4_row['TrackNumber'])
    else:
        return False, None

def has_time5_empty(output):
    status5_list = output[output.Status==5].index.tolist()
    time5_empty_row_number = status5_list[0] if len(status5_list) != 0 else None
    if time5_empty_row_number is not None:
        time5_row = output.loc[time5_empty_row_number]
        return True, transfer_string_to_int(time5_row['TrackNumber'])
    else:
        return False, None

def has_time6_empty(output):
    status6_list = output[output.Status == 6].index.tolist()
    time6_empty_row_number = status6_list[0] if len(status6_list) != 0 else None
    if time6_empty_row_number is not None:
        time6_row = output.loc[time6_empty_row_number]
        return True, time6_row['TrainNumber']
    else:
        return False, None

# video_station.release()
#get formate of this video, default is mp4
file_path = './data/21-22.mp4'
videoCapture = cv2.VideoCapture('./data/21-22.mp4')
time_start = file_path.split('/')[-1].split('-')[0]
time_start = time_start + ':00:00'
width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_data = pd.DataFrame(columns=['TrainNumber', 'Time1', 'Time2', 'Time3', 'Time4', 'Time5', 'Time6', 'TrackNumber', 'Status'])


#get dimension and bit rate
fps = videoCapture.get(cv2.CAP_PROP_FPS)
fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
print("[Information] Total frame number is: ", fNUMS)
# get frame
station_success, station_frame = videoCapture.read()
# fps = 59.99
time_video_size = (249, 237)
station_video_size = (2318, 660)
# time_video = cv2.VideoWriter("./data/17-18output_time_global_tt.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, time_video_size)
# station_video = cv2.VideoWriter("./data/17-18output_station_global_tt.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, station_video_size)

# crop video by two region, one is time record, another is the station
print("[Information] Cropping video, please wait...")
background = station_frame[326:986, 230:2548]
last_frame = station_frame[326:986, 230:2548]
curr_frame = 1
break_begin = False
for curr_frame in tqdm(range(2, int(fNUMS))):
    # print("当前是第 " + str(curr_frame) + "帧")
    # curr_frame += 1
    station_success, station_frame = videoCapture.read() # get next frame
    if curr_frame % 3 > 0:
        continue
    cropped_time_frame = station_frame[85:322, 15:264]
    cropped_station_frame = station_frame[326:986, 230:2548]

    diff = cv2.absdiff(background, cropped_station_frame)
    average = diff.mean(axis=0).mean(axis=0)
    if not has_train_coming(cropped_station_frame):#average[0] < 2 and average[1] < 2 and average[2] < 2 and not break_begin:
        background = cropped_station_frame
        last_frame = cropped_station_frame
        continue
    else:
        if has_new_train_comming(last_frame, cropped_station_frame):
            output_data = get_train_track_information(last_frame, cropped_station_frame, cropped_time_frame, output_data)
        # time1_empty, track_number = has_time1_empty(output_data)
        # if time1_empty:
        # time1, track_number = get_time1(last_frame, cropped_station_frame, cropped_time_frame, 1)
        # if time1 is not None:
        #     output_data = update_time1_information(time1, track_number, output_data)

        # time2_empty, track_number= has_time2_empty(output_data)
        # if time2_empty:
        #     time2 = get_time2(last_frame, cropped_station_frame, cropped_time_frame)
        #     if time2 is not None or time2 == '':
        #         # 去除重复的数据，假设同一秒没有同一班次的车来
        #         time2_list = output_data[output_data.Time2==time2].index.tolist()
        #         if time2 > time_start and len(time2_list) == 0:
        #             output_data = update_time2_information(time2, output_data)

        # 去除重复的数据，假设同一秒没有同一班次的车来
        # row_number = output_data[output_data.Time1==time1].index.tolist()[0]
        # if output.Time1.loc[row_number] == None:
        # time6_empty, train_number = has_time6_empty(output_data)
        # if time6_empty:
        #     time6 = get_time6(last_frame, cropped_station_frame, cropped_time_frame)
        #     if time6 is not None or time6 == '':
        #         if time6 > time_start:
        #             output_data = update_time6_information(time6, output_data, train_number)
        # if has_time2_empty(output_data):
        #     time2 = get_time2(last_frame, cropped_station_frame, cropped_time_frame)
        #     output_data = update_time2_information(time2, output_data)
        #     # get_train_track_information(last_frame, cropped_station_frame, cropped_time_frame, output_data)
        # if has_time3_empty(output_data):
        #     time3 = get_time3(last_frame, cropped_station_frame, cropped_time_frame)
        #     output_data = update_time3_information(time3, output_data)
        break_begin = True
        # output_data = get_train_track_information(last_frame, cropped_station_frame, cropped_time_frame, output_data)
        # get_train_number_region(frame)
        # get_time3(last_frame, frame, diff)
        # get_time1(last_frame, cropped_station_frame, cropped_time_frame)
        # get_time5(last_frame, frame, diff)
        # get_time2(last_frame, frame, diff)
        background = cropped_station_frame
        last_frame = cropped_station_frame

output_filename = './output/' + file_path.split('/')[-1].split('.')[0] + '.csv'
WriteExcel.write_data_to_csv(output_data, output_filename)
# time_video.release()
# station_video.release()
# output_data.add([])
videoCapture.release()