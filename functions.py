# -*- coding: utf-8 -*-

import abc
import numpy as np
import math

# 以下適当にいくつか最適化のベンチマーク関数を列挙する。
# 激しいのは割愛。
# classにする意味はないのかもしれない・・・

class Function:
	__metaclass__ = abc.ABCMeta

	def __call__(self):
		return self.function

	@abc.abstractmethod
	def function(self, x):
		raise NotImplementedError()

class AckleyFunction(Function):
	"""
		ackley's function
		search range
			x0 = [-32.768, -32.768]
			x1 = [-32.768, -32.768]
		global min : f(0, 0) = 0
	"""

	def function(self, x):
		# あんまり面白くない。
		return 20 - 20*np.exp(-0.2*np.sqrt(0.5*(x[0]**2 + x[1]**2))) - np.exp(0.5*(np.cos(2*math.pi*x[0])+np.sin(2*math.pi*x[1]))) + math.e
		
class AckleyFunction_mod(Function):
	def function(self, x):
		# 波々のところの調整版
		return 20 - 20*np.exp(-0.2*np.sqrt(0.5*(x[0]**2 + x[1]**2))) - np.exp(0.5*(np.cos(0.5*math.pi*x[0])+np.sin(0.5*math.pi*x[1]))) + math.e

class RosenbrockFunction(Function):
	"""
	Rosenbrock function 通称バナナ関数。
	search range
			x0 = [-5, 5]
			x1 = [-5, 5]
	global min : f(1,1) = 0
	"""

	def function(self, x):
		# うん。って感じ
		return 100*(x[1] - x[0]**2)**2 + (x[0]-1)**2

class BoothFunction(Function):
	"""
	Booth's function
	search range
		x0 = [-10, 10]
		x1 = [-10, 10]
	global min : f(1,3) = 0
	"""

	def function(self, x):
		# あんまりぱっとしない
		return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

class ThreeHumpCamelFunction(Function):
	"""
	Three-hump camel function
	search range
		x0 = [-5, 5]
		x1 = [-5, 5]
	global min : f(0,0) = 0
	"""

	def function(self, x):
		# グラフ全体見た感じだとわかりづらいけど一応波打って(0,0)に向かっているようだ。
		return 2*x[0]**2 - 1.05*x[0]**4 + x[0]**6/6 + x[0]*x[1] + x[1]**2

class McCormickFunction(Function):
	"""
	McCormick function
	search range
		x0 = [-1.5, 4]
		x1 = [-3, 4]
	global min : f(-0.54791,-1.54719) = 0
	"""

	def function(self, x):
		# そうだな。って感じ。
		return np.sin(x[0] + x[1]) + (x[0] - x[1])**2 -1.5*x[0] + 2.5*x[1] + 1


class FiveWellPotentialFunction(Function):
	"""
	five-well potential function
	search range
		x0 = [-20, 20]
		x1 = [-20, 20]
	global min : f(4.92, -9:89) = -1.4616
	"""

	def function(self, x):
		# 微妙に図と違うのが出来上がってるのはなぜなのかよく分からない。
		return (1.0 -1.0/(1.0+0.05*(x[0]**2 + (x[1]-10))**2)
					-1.0/(1.0+0.05*((x[0]-10)**2 + x[1]**2))
					-1.5/(1.0+0.03*((x[0]+10)**2 + x[1]**2))
					-2.0/(1.0+0.05*((x[0]-5)**2 + (x[1]+10)**2))
					-1.0/(1.0+0.1*((x[0]+5)**2 + (x[1]+10)**2))
				) * (1.0+0.0001*(x[0]**2 + x[1]**2)**1.2)

class FiveWellPotentialFunction_mod(Function):		
	# five-wellちょっといじったらいい感じのになったのでの残しておく。
	# 個人的には一番オススメ。絵に描いたような複数の局所解のあるグラフ

	def function(self, x):
		return (1.0 -1.0/(1.0+0.05*(x[0]**2 + (x[1]-10)**2))
					-1.0/(1.0+0.05*((x[0]-10)**2 + x[1]**2))
					-1.5/(1.0+0.03*((x[0]+10)**2 + x[1]**2))
					-2.0/(1.0+0.05*((x[0]-5)**2 + (x[1]+10)**2))
					-1.0/(1.0+0.1*((x[0]+5)**2 + (x[1]+10)**2))
				) * (1.0+0.0001*(x[0]**2 + x[1]**2)**1.2) +1.5 #ここの1.5でちょっと調整
