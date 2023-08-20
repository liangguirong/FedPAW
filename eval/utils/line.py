# collections模块还提供了几个额外的数据类型：Counter、deque、defaultdict、namedtuple和OrderedDict等。
# 1.namedtuple: 生成可以使用名字来访问元素内容的tuple
# 2.deque: 双端队列，可以快速的从另外一侧追加和推出对象
# 3.OrderedDict: 有序字典
# 4.defaultdict: 带有默认值的字典
# 5.Counter: 计数器，主要用来计数

from typing import Tuple, Iterable, Callable, Collection, List
# 要定义一个类型别名，可以将一个类型赋给别名。类型别名可用于简化复杂类型签名，在下面示例中，Vector 和 list[float] 将被视为可互换的同义词：
# Vector = list[float]
Point = Collection[float]

class Line:
    # 将形参给到pbeg和pend，同时检查他们是否是Point类类型，相当于强制类型检查
    def __init__(self, pbeg:Point, pend:Point):
        self.pbeg = pbeg
        self.pend = pend
        # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
        # zip([iterable, ...])
        # >>> a = [1,2,3]
        # >>> b = [4,5,6]
        # >>> c = [4,5,6,7,8]
        # >>> zipped = zip(a,b)     # 打包为元组的列表
        # [(1, 4), (2, 5), (3, 6)]
        # >>> zip(a,c)              # 元素个数与最短的列表一致
        # [(1, 4), (2, 5), (3, 6)]
        # 这样的话就仅照顾距离最短的线，并求出了差集
        self.pdlt = [(ev - bv) for bv, ev in zip(self.pbeg, self.pend)]


    def at(self, x:float) -> Point:
        # lt是差集，用x乘以差集再加上第一条直线
        return tuple(bv + x * dv for bv, dv in zip(self.pbeg, self.pdlt))


# 由线建立曲线的类
class CurveByLines(Line):
    def __init__(self, pts: Collection[Point]):
        # 自身属性
        self.lines = [Line(pbeg, pend) for pbeg, pend in zip(pts, pts[1:])]

    # 此处对Curve类进行分步处理，x为一个比例，处在[0，1]区间
    def at(self, x: float) -> Point:
        # 首先按比例找到对应的lines索引
        x_with_lines = x * len(self.lines)
        # 找到离索引最近的实际坐标点索引值，且不能超出最大索引范围
        li = min(int(x_with_lines), len(self.lines) - 1)
        # print(self.lines[li].at(x_with_lines - li))
        # 此处返回的其实就是分好段的离散坐标了
        return self.lines[li].at(x_with_lines - li)

class CurveByFormula(Line):
    def __init__(self, f: Callable[[float], Point]):
        self.f = f

    def at(self, x:float) -> Point:
        return self.f(x)

def discretize(line:Line, n:int, xbeg:float=0.0, xend:float=1.0) -> Iterable[Point]:
    # 此处的line为curve，是一个line的集合
    xinv = xend - xbeg
    # 即：此处的at应参考curve类中的at方法
    return (line.at((i / n) * xinv + xbeg) for i in range(0, n+1))
