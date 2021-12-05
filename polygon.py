import cv2
from typing import Union, List, Tuple


def is_ray_intersects_segment(point: Union[List, Tuple],
                              start_p: Union[List, Tuple],
                              end_p: Union[List, Tuple]) -> bool:  # [x,y] [lng,lat]
    # 输入：判断点，边起点，边终点，都是[lng,lat]格式数组
    if start_p[1] == end_p[1]:  # 排除与射线平行、重合，线段首尾端点重合的情况
        return False
    if start_p[1] > point[1] and end_p[1] > point[1]:  # 线段在射线上边
        return False
    if start_p[1] < point[1] and end_p[1] < point[1]:  # 线段在射线下边
        return False
    if start_p[1] == point[1] and end_p[1] > point[1]:  # 交点为下端点，对应spoint
        return False
    if end_p[1] == point[1] and start_p[1] > point[1]:  # 交点为下端点，对应epoint
        return False
    if start_p[0] < point[0] and end_p[0] < point[0]:  # 线段在射线左边
        return False

    xseg = end_p[0] - (end_p[0] - start_p[0]) * (end_p[1] - point[1]) / (end_p[1] - start_p[1])  # 求交

    # 交点在射线起点的左侧
    return False if xseg < point[0] else True
    # if xseg < point[0]:  # 交点在射线起点的左侧
    #     return False
    # return True  # 排除上述情况之后


def crossing_number(point: Union[List, Tuple], polys: List) -> bool:
    """
    point:(x,y)
    poly: 闭合线圈集合:[[(x1,y1),(x2,y2),...],[],...]
    """

    cn = 0  # 交点个数
    for poly in polys:
        poly_len = len(poly)
        limit = lambda x: x % poly_len

        # for i in range(poly_len - 1):  # [0,len-1]
        for i in range(len(poly)):
            # start_p = poly[i]
            # end_p = poly[i + 1]

            start_p = poly[limit(i)]
            end_p = poly[limit(i + 1)]
            if is_ray_intersects_segment(point, start_p, end_p):
                cn += 1  # 有交点就加1
    # print('cn:',cn)
    # outside=0
    # inside=1
    return True if cn % 2 == 1 else False


def is_left(p0: Union[List, Tuple], p1: Union[List, Tuple], p2: Union[List, Tuple]) -> int:
    # >0 for p2 left of the poly through p0 and p1
    # =1 for p2  ont he poly
    # <2 for p2 right of the poly
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])


def winding_number(p: Union[List, Tuple], poly: List) -> bool:
    """
    wn=ths winging number(=0 only when p is outside)
    """
    wn = 0
    poly_len = len(poly)
    limit = lambda x: x % poly_len
    for i in range(poly_len):

        start_p = poly[limit(i)]
        end_p = poly[limit(i + 1)]

        if start_p[1] <= p[1]:  # start y<=p.y
            if end_p[1] > p[1]:  # an upward crossing
                if is_left(start_p, end_p, p) > 0:  # p left of edge
                    wn += 1

        else:
            if end_p[1] <= p[1]:  # a downward crossing
                if is_left(start_p, end_p, p) < 0:  # p right of edge
                    wn -= 1
    print(f'wn:{wn}')
    return False if wn == 0 else True


if __name__ == '__main__':

    poly = [(415, 210),
            (552, 227),
            (710, 255),
            (684, 204),
            (715, 186),
            (602, 152),
            (517, 192)
            ]
    # p = (500, 200)
    # p = (100, 100)
    p = (600, 200)
    img = cv2.imread('w.png')
    for i, data in enumerate(poly):
        x = lambda x, len: x % len
        cv2.line(img, poly[x(i, len(poly))], poly[x(i + 1, len(poly))], (0, 0, 255), 2)
    cv2.circle(img, p, 2, (0, 0, 255), -1, -1)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    flag = crossing_number(p, [poly])
    print(flag)
    # r = crossing_number(p, poly)
    # wn = winding_number(p, poly)
    # print(r)
    # print(wn)
