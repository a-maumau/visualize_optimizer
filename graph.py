# -*- coding: utf-8 -*-

import numpy as np

# グラフ
class Graph:
	def __init__(self, range_x, range_y, func=None):
		self.f = func if func is not None else self.default_function # ソースいじらずに自分で設定したい人用に準備しときました。
		self.xr, self.yr = np.meshgrid(range_x, range_y) 			 # センス的にはx1, x2に合わせるべきか。
		self.z = self.f([self.xr, self.yr]) 			 			 # ただここのzをfと書くと被るしで微妙なので...

	def default_function(self, x):
		#とりあえずこれって感じ。
		return 1/10.0*x[0] ** 2 + 2*x[1] ** 2

	def mesh(self):
		return [self.xr, self.yr, self.z]
