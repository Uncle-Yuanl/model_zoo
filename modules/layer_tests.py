import unittest
# import sys
# sys.path.append('/home/yuanhao/model_zoo')

from modules.layers import *

class TestConcatenate1D(unittest.TestCase):
    a = tf.constant([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    b = tf.constant([[[10, 20, 30], [40, 50, 60], [70, 80, 90]]])
    c1d = Concatenate1D(name='test')
    def test3d(self):
        """这个应该要通过"""
        c3 = self.c1d((self.a, self.b))
        print(c3)

    def test2d(self):
        """这是基础使用, 也只有这个通过了"""
        c2 = self.c1d((self.a[0], self.b[0]))
        print(c2)

    def test1d(self):
        """axis=1导致无法成功，axis=0可以的"""
        c1 = self.c1d((self.a[0][0], self.b[0][0]))
        print(c1)


class TestBERT(unittest.TestCase):
    pass





if __name__ == '__main__':
    unittest.main()
    # 指定类运行单元测试
    class2run = [TestConcatenate1D]

    loader = unittest.TestLoader()
    suits = []
    for cls in class2run:
        suits.append(loader.loadTestsFromTestCase(cls))
    suits = unittest.TestSuite(suits)

    runner = unittest.TestRunner()
    results = runner.run(suits)