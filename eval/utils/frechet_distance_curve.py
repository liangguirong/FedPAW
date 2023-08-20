import line
import frechet
def test_2d_curve(c1, c2):
    # 建立CurveByLines类，a，b具有一定的属性、方法了
    # 具体包含：线集，差集，at方法
    a = line.CurveByLines(c1)
    b = line.CurveByLines(c2)
    # 标准化格式后的线，进行弗雷歇距离计算，a，b是一个可迭代的对象
    return frechet.frechet_distance(a, b)
