import cv2
import time
from picam import Imget
import numpy as np
import heapq
from global_var import gol

global blue_lower, blue_upper, red_lower, red_upper, blue_cx, blue_cy, red_cx, red_cy, x_units, y_units, next_0, next_1, matrix

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

class proc:
    def __init__(self) -> None:
        pass
    # 处理图像
    def process_image(self, image):
        global blue_lower, blue_upper, red_lower, red_upper, blue_cx, blue_cy, red_cx, red_cy, x_units, y_units, next_0, next_1, matrix
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 蓝色色块检测
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 红色色块检测
        red_mask = cv2.inRange(hsv, red_lower, red_upper)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 寻找色块中心点并绘制
        for contour in blue_contours:
            area = cv2.contourArea(contour)
            if area > 100:
                M = cv2.moments(contour)
                blue_cx = int(M['m10'] / M['m00'])
                blue_cy = int(M['m01'] / M['m00'])
                cv2.circle(image, (blue_cx, blue_cy), 5, (255, 0, 0), -1)
                # 数据共享
                gol().set_value(blue_cx, blue_cx)
                gol().set_value(blue_cy, blue_cy)
        for contour in red_contours:
            area = cv2.contourArea(contour)
            if area > 100:
                M = cv2.moments(contour)
                red_cx = int(M['m10'] / M['m00'])
                red_cy = int(M['m01'] / M['m00'])
                cv2.circle(image, (red_cx, red_cy), 5, (0, 0, 255), -1)
                gol().set_value(red_cx, red_cx)
                gol().set_value(red_cy, red_cy)
        x_units = int((red_cx-blue_cx)/11)# 红色色块中心x坐标至蓝色色块中心x坐标像素点11等分平均值，作为单位量
        y_units = int((blue_cy-red_cy)/9)# 红色色块中心y坐标至蓝色色块中心y坐标像素点9等分平均值，作为单位量
        gol().set_value(x_units, x_units)
        gol().set_value(y_units, y_units)
        return image


    def get_perspective_transform(self,image):
        global blue_lower, blue_upper, red_lower, red_upper, blue_cx, blue_cy, red_cx, red_cy, x_units, y_units, next_0, next_1, matrix
        # 定义源点（感兴趣区域的四个角点）
        src = np.float32([(blue_cx-x_units, red_cy-y_units), (blue_cx-x_units, blue_cy+y_units), (red_cx+x_units, red_cy-y_units), (red_cx+x_units, blue_cy+y_units)])

        # 定义目标点（映射感兴趣区域的坐标）
        dst = np.float32([[0, 0], [0, 360], [480, 0], [480, 360]])

        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(src, dst)

        # 应用透视变换到图像
        warped_image = cv2.warpPerspective(image, M, (480, 360))
        return warped_image

    #路径规划
    def get_neighbors(self, node, matrix):
        global blue_lower, blue_upper, red_lower, red_upper, blue_cx, blue_cy, red_cx, red_cy, x_units, y_units, next_0, next_1
        """
        获取节点的相邻节点
        """
        neighbors = []
        rows, cols = matrix.shape
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # 右、左、下、上
        for dx, dy in directions:
            x, y = node
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and matrix[nx, ny] != 2:
                neighbors.append((nx, ny))
        return neighbors

    def heuristic(self, node, goal):
        """
        估计从当前节点到目标节点的代价（欧几里得距离）
        """
        return np.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

    def a_star(self, matrix, start, goal):
        """
        A*算法实现路径规划
        """
        rows, cols = matrix.shape
        visited = set()
        open_list = [(0, start)]  # 优先队列，元素为 (f, node)
        g_score = {start: 0}  # 起点到节点的实际代价
        f_score = {start: self.heuristic(start, goal)}  # 起点经过节点到目标节点的估计代价
        came_from = {}  # 记录每个节点的前驱节点

        while open_list:
            _, current = heapq.heappop(open_list)
            if current == goal:
                # 找到了路径，回溯重构路径
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            visited.add(current)

            neighbors = self.get_neighbors(current, matrix)
            for neighbor in neighbors:
                g = g_score[current] + 1  # 两个相邻节点的距离为1
                if neighbor in visited or g >= g_score.get(neighbor, np.inf):
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = g
                f_score[neighbor] = g + self.heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score[neighbor], neighbor))

        # 无法到达终点，返回空路径
        return []

    def planning (self, matrix, start, goal):
        path = []
        start_x,start_y = start
        goal_x,goal_y = goal
        if (20>start_x>0) and (20 > start_y > 0)and (20>goal_x>0) and (20>goal_y>0):
            matrix[goal_x][goal_y] = 1
            path = self.a_star(matrix, start, goal)
            matrix[goal_x][goal_y] = 2
            return path
        else :
            return []

    #(9,5)->(11,17)
    def right_conversion(self, x1,y1):
        y2 = 2*x1-1
        x2 = 21-2*y1
        return x2,y2
    #(11,17)->(9,5)
    def left_conversion(self, x2,y2):
        y1 = (21-x2)/2
        x1 = (y2+1)/2
        return x1,y1

    def dots_find(self, img):
        if img is None:
            return
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 高斯模糊降噪
        gray_img = cv2.GaussianBlur(gray_img, (9, 9), 2, 2)
        circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, dp = 1, minDist = 20,
                                param1 = 50, param2 = 30, minRadius = 0, maxRadius=0)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(img, (x, y), r, (0, 255, 0), 4)
                cv2.circle(img, (x, y), 1, (0, 0, 255), 3)

            return circles, img
        else:
            print ("we got sth. wrong")

    #设置列表比较
    def takeSecond(self, elem):
        return elem[1]# 处理图像