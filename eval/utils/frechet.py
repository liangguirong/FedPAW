from typing import Any, Callable, List
from shapesimilarity import shape_similarity
import numpy as np
import matplotlib.pyplot as plt
import line

# 弗雷歇距离求和
def transform(l):
    return [list(el) for el in l]

def similarity(l1, l2):

    r1 = transform(l1)
    # print("r1:{}".format(r1))
    r2 = transform(l2)

    shape1 = np.row_stack(r1)
    shape2 = np.row_stack(r2)

    # print("shape1:{}".format(shape1))

    similarity = shape_similarity(shape1, shape2)

    # print("similartiy:{}".format(similarity))

    # print(similarity)

    # plt.plot(shape1[:, 0], shape1[:, 1], linewidth=2.0)
    # plt.plot(shape2[:, 0], shape2[:, 1], linewidth=2.0)
    # plt.title(f'Shape similarity is: {similarity}', fontsize=14, fontweight='bold')
    # plt.show()
    return similarity

def frechet_distance(
    # ->常常出现在python函数定义的函数名后面，为函数添加元数据,描述函数的返回类型，也可以理解为给函数添加注解
    # 形参后面加冒号其实是添加注释，告诉使用者每个形参、返回值的类型，这里只是建议，传入其他类型也并不会报错
    # 此处若传入的为曲线，则接受的为CurveByLines类
    l1: line.Line,
    l2: line.Line,
    # 节点数，离散化阈值
    n_disc_l1: int = 100,
    n_disc_l2: int = 100,
    # 星号将多个实参合并为一个元组
    *,
    # 预设的阈值
    prec: float = 0.001,

) -> float:
    # print(l1)
    # 输入中，l1、l2为曲线，n_xx为分段数
    return distance_matrix(l1, l2, n_disc_l1, n_disc_l2)

# 离散化，求距离矩阵（二维数组） 距离矩阵
def distance_matrix(l1:line.Line, l2:line.Line, nd1:int, nd2:int) -> List[List[float]]:

    # 此时的l1于l2依旧是curve类
    # 保留曲线的起点，在后面将其细分为参数份
    ld1 = list(line.discretize(l1, nd1))
    # print(ld1)
    ld2 = list(line.discretize(l2, nd2))
    return similarity(ld1, ld2)
