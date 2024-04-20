class PassFinder:
    def __init__(self) -> None:
        pass
    def generate(self, path):
        dot_pre = ()
        director = 0
        road = [];
        for dot in path:
            dot_pre = dot
            if(dot_pre == dot):
                continue
def turn_direction(self, coord1, coord2, coord3):
    x1, y1 = coord1
    x2, y2 = coord2
    x3, y3 = coord3
    # 计算向量叉乘
    cross_product = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
    if cross_product > 0:
        return "l"
    elif cross_product < 0:
        return "r"
    else:
        return "d"