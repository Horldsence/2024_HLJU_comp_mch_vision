import cv2
import time
from picam import Imget
import numpy as np
import torch
from bluetooth_ import MyBluetooth
import bluetooth
from processor import proc
from global_var import gol

getImg = Imget()

global blue_lower
global blue_upper
global red_lower
global red_upper

global blue_cx
global blue_cy
global red_cx
global red_cy
global x_units
global y_units
global next_0
global next_1

global matrix

blue_lower = np.array([100, 50, 50])
blue_upper = np.array([130, 255, 255])
red_lower = np.array([0, 50, 50])
red_upper = np.array([10, 255, 255])

blue_cx=0
blue_cy=0
red_cx=0
red_cy=0
x_units=0
y_units=0
next_0 = False
next_1 = False

#地图矩阵
#1表示通路，2表示障碍，3表示起点，4表示终点
#matrix[y][x]-----注意和宝藏图坐标（x，y）一致，xy为宝藏图坐标
matrix = np.array([
    #0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10,11,12,13,14,15,16,17,18,19,20----x
    [2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2, 2, 2, 2 ,2 ,2],#0
    [2 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,1 ,1 ,1],#1
    [2 ,1 ,2 ,1 ,2 ,2 ,2 ,1 ,2 ,1 ,2 ,1 ,2 ,2 ,2 ,1 ,2 ,1 ,2 ,2 ,2],#2
    [2 ,1 ,2 ,1 ,2 ,1 ,1 ,1 ,2 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,2 ,1 ,1 ,1 ,2],#3
    [2 ,1 ,2 ,2 ,2 ,1 ,2 ,2 ,2 ,1 ,2 ,2 ,2 ,1 ,2 ,1 ,2 ,2 ,2 ,1 ,2],#4
    [2 ,1 ,2 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,2 ,1 ,1 ,1 ,2],#5
    [2 ,2 ,2 ,1 ,2 ,1 ,2 ,1 ,2 ,2 ,2 ,2 ,2 ,1 ,2 ,2 ,2 ,1 ,2 ,2 ,2],#6
    [2 ,1 ,1 ,1 ,2 ,1 ,2 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,2],#7
    [2 ,1 ,2 ,2 ,2 ,1 ,2 ,1 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,1 ,2 ,2 ,2 ,1 ,2],#8
    [2 ,1 ,2 ,1 ,1 ,1 ,2 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,2],#9
    [2 ,1 ,2 ,2 ,2 ,2 ,2 ,1 ,2 ,2 ,2 ,2 ,2 ,1 ,2 ,2 ,2 ,2 ,2 ,1 ,2],#10
    [2 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,1 ,1 ,2 ,1 ,2],#11
    [2 ,1 ,2 ,2 ,2 ,1 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,1 ,2 ,1 ,2 ,2 ,2 ,1 ,2],#12
    [2 ,1 ,2 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,2 ,1 ,1 ,1 ,2],#13
    [2 ,2 ,2 ,1 ,2 ,2 ,2 ,1 ,2 ,2 ,2 ,2 ,2 ,1 ,2 ,1 ,2 ,1 ,2 ,2 ,2],#14
    [2 ,1 ,1 ,1 ,2 ,1 ,2 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,2],#15
    [2 ,1 ,2 ,2 ,2 ,1 ,2 ,1 ,2 ,2 ,2 ,1 ,2 ,2 ,2 ,1 ,2 ,2 ,2 ,1 ,2],#16
    [2 ,1 ,1 ,1 ,2 ,1 ,2 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,1 ,1 ,2 ,1 ,2 ,1 ,2],#17
    [2 ,2 ,2 ,1 ,2 ,1 ,2 ,2 ,2 ,1 ,2 ,1 ,2 ,1 ,2 ,2 ,2 ,1 ,2 ,1 ,2],#18
    [1 ,1 ,1 ,1 ,2 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,2],#19
    [2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2, 2, 2, 2 ,2 ,2] #20-----y
])

path = proc().planning(matrix, (1, 19), (19, 1))
print(path)

photo_get_judge = 0
start_time = time.time()

if __name__ == '__main__':
    # 主程序
    while open:
        image = getImg.getImg()
        if image is None:
            break
        # 处理图像
        processed_image = proc().process_image(image)
        # 显示图像和坐标
        new_image = proc().get_perspective_transform(processed_image)
        new_image = proc().process_image(new_image)
        line_drew_image = np.copy(new_image)
        # 获取更新后数据
        blue_cx = gol().get_value(blue_cx)
        blue_cy = gol().get_value(blue_cy)
        red_cx = gol().get_value(red_cx)
        red_cy = gol().get_value(red_cy)
        x_units = gol().get_value(x_units)
        y_units = gol().get_value(y_units)
        # 绘制直线确保拍摄准确
        cv2.line(processed_image,(blue_cx-x_units, red_cy-y_units),(blue_cx-x_units, blue_cy+y_units),(0, 0, 0),3)
        cv2.line(processed_image,(blue_cx-x_units, red_cy-y_units),(red_cx+x_units, red_cy-y_units),(0, 0, 0),3)
        cv2.line(processed_image,(red_cx+x_units, blue_cy+y_units),(red_cx+x_units, red_cy-y_units),(0, 0, 0),3)
        cv2.line(processed_image,(red_cx+x_units, blue_cy+y_units),(blue_cx-x_units, blue_cy+y_units),(0, 0, 0),3)
        for b in range(0,11):#横
            cv2.line(line_drew_image,((blue_cx+int(x_units/2)),(red_cy-int(y_units/2))+(b*y_units)),((red_cx-int(x_units/2)), (red_cy-int(y_units/2))+(b*y_units)),(0, 0, 0),3)
        for v in range(0,11):#竖
            cv2.line(line_drew_image,((blue_cx+int(x_units/2))+(v*x_units),(blue_cy+int(y_units/2))),((blue_cx+int(x_units/2))+(v*x_units), (red_cy-int(y_units/2))),(0, 0, 0),3)
        cv2.putText(line_drew_image,"'Q'->got_photo",(10,40),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,255,255),2)

        # results = model(new_image)
        # results_ = results.pandas().xyxy[0].to_numpy()
        # circles, doted_proc_img = dots_find(processed_image)
        # cv2.imshow("doted_img", doted_proc_img)

        cv2.namedWindow('Image',0)
        cv2.namedWindow('NewImage',0)
        cv2.namedWindow('line_drew_image',0)
        cv2.imshow("Image", processed_image)
        cv2.imshow("NewImage",new_image)
        cv2.imshow("line_drew_image",line_drew_image)
        cv2.resizeWindow('NewImage',500,320)
        cv2.resizeWindow('Image',640,360)
        cv2.resizeWindow('line_drew_image',500,320)
        cv2.moveWindow('Image',0,0)
        cv2.moveWindow('newimage',650,0)
        cv2.moveWindow('line_drew_image',650,400)
        print(x_units,y_units)

        if cv2.waitKey(1) == ord('q'):
            cv2.imwrite('~/Desktop/test/image/saved_image.jpg', new_image)
            image_path = '~/Desktop/test/image/saved_image.jpg'
            test_image = cv2.imread(image_path)
            if test_image is not None:
                cv2.destroyAllWindows()
                cv2.imshow('Image_test', test_image)
                cv2.waitKey(5000)
                next_0 = True
                break
    cv2.destroyAllWindows()

    model = torch.hub.load('~/yolov5','custom',path='~/文档/block_yolo5/last.pt',source='local')#加载yolov5训练的模型
    model.conf = 0.35#设置置信度

    blue_cx=0
    blue_cy=0
    red_cx=0
    red_cy=0

    Treasure_coordinates = []#宝藏临时坐标
    Precious_Dot_path = []#宝藏坐标
    map_path = []#宝藏的地图坐标
    plan_path = []

    #调试用
    while True:
    #整体流程用
    #while next_0:
        image_path = '~/Desktop/test/image/saved_image.jpg'
        image = cv2.imread(image_path)
        results = model(image)
        # 处理预测结果
        # 提取预测框、类别和置信度等信息
        results_ = results.pandas().xyxy[0].to_numpy()
        image = proc().process_image(image)
        i = 0

        #预测框描绘
        for box in results_:
            l,t,r,b = box[:4].astype('int')
            confidence = str(round(box[4]*100,2))+"%"
            cls_name = box[6]

            #宝藏坐标转换---Treasure_coordinates[]
            center_x = (l+r)/2+5  # 获取圆心 x 坐标
            center_y = (t+b)/2-5 # 获取圆心 y 坐标
            converted_x = int((center_x-blue_cx+ 0)/x_units)
            converted_y = int((blue_cy-center_y- 0)/y_units)+1
            Treasure_coordinates.append((converted_x, converted_y))
            if converted_x >= 0 and converted_y <= 9:
                Precious_Dot_path.append((converted_x, converted_y))
            else :
                pass

            #标注block
            if cls_name == "Precious_Dot":
                i += 1
            else :
                pass
            cv2.rectangle(image,(l,t),(r,b),(0,255,0),2)
            cv2.putText(image,"(" + str(converted_x) + "," + str(converted_y) + ")",(l,t),cv2.FONT_ITALIC,1,(255,0,0),2)
        #Precious_Dot_path[0]----起点
        #Precious_Dot_path[9]----终点
        # del Treasure_coordinates[8:],Precious_Dot_path_l_x[2:],Precious_Dot_path_r_x[2:],Precious_Dot_path_l_s[2:],Precious_Dot_path_r_s[2:],Precious_Dot_path[9:]
        Precious_Dot_path.append((3,4))
        # Precious_Dot_path = Precious_Dot_path + Precious_Dot_path_l_x + Precious_Dot_path_r_x + Precious_Dot_path_r_s + Precious_Dot_path_l_s
        Precious_Dot_path.append((9,8))
        #print("Precious_Dot_path",Precious_Dot_path)

        #宝藏图坐标转换地图坐标
        for a in Precious_Dot_path:
            x1,y1 = a
            x2,y2 = proc().right_conversion(x1,y1)
            map_path.append((x2,y2))
            #Precious_Dot_path.clear()

        cv2.imwrite('~/Desktop/test/image/new_image.jpg', image)
        print(map_path)
        print(len(map_path))

        '''
        print("map_path_l_x",map_path_l_x)
        print("map_path_r_x",map_path_r_x)
        print("map_path_r_s",map_path_r_s)
        print("map_path_l_s",map_path_l_s)
        '''
        Precious_Dot_path.clear()


        for plan in map_path:
            c,d =plan
            if (c == 13 and d == 5) or (c == 7 and d == 17):
                continue
            matrix[c][d] = 2

        # 显示图像
        cv2.imwrite('~/Desktop/test/image/new_image.jpg', image)
        cv2.putText(image,"Part2",(5,30),cv2.FONT_HERSHEY_COMPLEX,1.2,(0,255,255),1)
        cv2.imshow('Image', image)
        cv2.waitKey(5000)
        next_1 = True
        break
        '''
        if cv2.waitKey(1) == 27:
            break
        '''
    cv2.destroyAllWindows()

    SendData = []              #蓝牙发送数据，0,1,7数据为握手数据
    MyBluetooth.FindDevices();

    # 指定蓝牙设备的MAC地址和服务UUID
    device_address = 'XX:XX:XX:XX:XX:XX'  # 替换为您要连接的蓝牙设备的MAC地址
    service_uuid = 'XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX'  # 替换为您要连接的蓝牙设备的服务UUID

    # 连接蓝牙设备
    Sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    Sock.connect((device_address, 1))  # 1 是RFCOMM端口号，默认为1

    SendData = ['0x88']
    MyBluetooth.SendData(Sock, SendData)

    # 接收蓝牙设备的响应
    ReceiveData = MyBluetooth.ReceiveData(Sock)
    print('Received:', ReceiveData)

    path = proc().planning (matrix, (0, 19), (20, 1))
    if path is []:
        print("path generated error!")

    while True:
        image_path = '~/Desktop/test/image/saved_image.jpg'
        image = cv2.imread(image_path)
        cv2.namedWindow('Road')
        cv2.imshow("Road", image)
        # path = planning (matrix, (0, 19), (20, 1))
