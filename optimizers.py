# -*- coding: utf-8 -*-

import numpy as np
import abc
# 数値微分の微小区間値
delta = 1e-4

# 最適化手法
class Optimizer(object):
	__metaclass__ = abc.ABCMeta

	def __init__(self, f, init_pos, learning_rate=0.01, name=None, color="red"):
		self.f = f
		self.learning_rate = learning_rate
		self.x = init_pos
		self.next_pos = init_pos
		self.gradient = np.zeros_like(init_pos)

		self.name = name
		self.color = color

	def numerical_diff(self, x, i):
		"""
		#中央差分・数値微分
		#i : 偏微分する変数のインデックス
		"""
		h_vec = np.zeros_like(x)
		h_vec[i] = delta

		# 数値微分を使って偏微分する
		return (self.f(x + h_vec) - self.f(x - h_vec)) / (2.0 * delta)

	@abc.abstractmethod
	def optimize(self):
		"""
		各optimizerに合わせた実装に。
		一応gradientの表示等のことを考えて、
		gradientに勾配値を、next_posには次の更新点 pos - gradの結果を入れる形で。
		"""
		raise NotImplementedError()
		
	def pos(self):
		pos = np.concatenate((self.x, [self.f(self.x)]))
		return pos

	def next_pos(self):
		next_pos = np.concatenate((self.next_pos, [self.f(self.next_pos)]))
		return next_pos

	def pos_gradient(self):
		return self.gradient

	def update(self):
		self.x = self.next_pos
		self.optimize()

# GD
class GDOptimizer(Optimizer):
	def optimize(self):
		for i, _ in enumerate(self.x):
			# i 番目の変数で偏微分する
			self.gradient[i] = -self.numerical_diff(self.x, i)*self.learning_rate

		# = grad#-grad*self.learning_rate
		# 更新
		self.next_pos = self.x + self.gradient

# 擬似的なSGD
class SGDOptimizer(Optimizer):
	def __init__(self, f, init_pos, learning_rate=0.01, momentum=0.9, noize_vec_mul=2.0, noize_vec_negbias=0.3, noize_const_mul=1.0, noize_const_negbias=0.5,name=None, color="red"):
		super(SGDOptimizer, self).__init__(f, init_pos, learning_rate, name, color)
		self.noize_vec_mul = noize_vec_mul
		self.noize_vec_negbias = noize_vec_negbias
		self.noize_const_mul = noize_const_mul
		self.noize_const_negbias = noize_const_negbias
	
	def optimize(self):
		# 勾配を入れるベクトルをゼロで初期化する
		grad = np.zeros_like(self.x)

		for i, _ in enumerate(self.x):
			# i 番目の変数で偏微分する + ノイズを入れる。
			self.gradient[i] = self.numerical_diff(self.x, i)*(np.random.random()*self.noize_vec_mul - self.noize_vec_negbias) + (np.random.random()*self.noize_const_mul - self.noize_const_negbias)

		# 更新
		self.next_pos = self.x  -self.gradient*self.learning_rate

# Momentum
class MomentumOptimizer(Optimizer):
	def __init__(self, f, init_pos, learning_rate=0.01, momentum=0.9, name=None, color="red"):
		super(MomentumOptimizer, self).__init__(f, init_pos, learning_rate, name, color)
		self.momentum = momentum

	def optimize(self):
		# 勾配を入れるベクトルをゼロで初期化する
		grad = np.zeros_like(self.x)

		for i, _ in enumerate(self.x):
			# i 番目の変数で偏微分する
			grad[i] = -self.numerical_diff(self.x, i)*self.learning_rate

		self.gradient = self.momentum*self.gradient + grad
		# 更新
		self.next_pos = self.x + self.gradient

# Nesterov Accelerated Gradient
class NAGOptimizer(Optimizer):
	def __init__(self, f, init_pos, learning_rate=0.01, momentum=0.9, name=None, color="red"):
		super(NAGOptimizer, self).__init__(f, init_pos, learning_rate, name, color)
		self.momentum = momentum

	def optimize(self):
		# 勾配を入れるベクトルをゼロで初期化する
		grad = np.zeros_like(self.x)

		self.x -= self.gradient
		for i, _ in enumerate(self.x):
			# i 番目の変数で偏微分する
			grad[i] = self.numerical_diff(self.x, i)*self.learning_rate

		self.gradient = self.momentum*self.gradient + grad
		# 更新
		self.next_pos = self.x  -self.gradient

# AdaGrad
class AdaGradOptimizer(Optimizer):
	def __init__(self, f, init_pos, learning_rate=0.01, eps=1e-7, name=None, color="red"):
		super(AdaGradOptimizer, self).__init__(f, init_pos, learning_rate, name, color)
		self.h = np.zeros_like(init_pos)
		self.eps = eps

	def optimize(self):
		for i, _ in enumerate(self.x):
			# i 番目の変数で偏微分する
			self.gradient[i] = self.numerical_diff(self.x, i)

		self.h = self.h + self.gradient*self.gradient
		# 更新
		self.next_pos = self.x - self.gradient*self.learning_rate/np.sqrt(self.h+self.eps)

# RMSprop
class RMSpropOptimizer(Optimizer):
	def __init__(self, f, init_pos, learning_rate=0.01, alpha=0.99, eps=1e-7, name=None, color="red"):
		super(RMSpropOptimizer, self).__init__(f, init_pos, learning_rate, name, color)
		self.h = np.zeros_like(init_pos)
		self.alpha = alpha
		self.alpha_ = 1.0 - alpha
		self.eps = eps

	def optimize(self):
		for i, _ in enumerate(self.x):
			# i 番目の変数で偏微分する
			self.gradient[i] = self.numerical_diff(self.x, i)

		self.h = self.alpha*self.h + self.alpha_*self.gradient*self.gradient
		# 更新
		self.next_pos = self.x -self.gradient*self.learning_rate/(np.sqrt(self.h)+self.eps)

# RMSprop
class RMSpropMomentumOptimizer(Optimizer):
	def __init__(self, f, init_pos, learning_rate=0.01, alpha=0.99, momentum=0.9, eps=1e-7, name=None, color="red"):
		super(RMSpropMomentumOptimizer, self).__init__(f, init_pos, learning_rate, name, color)
		self.h = np.zeros_like(init_pos)
		self.alpha = alpha
		self.alpha_ = 1.0 - alpha
		self.momentum = momentum
		self.eps = eps

	def optimize(self):
		# 勾配を入れるベクトルをゼロで初期化する
		_grad = np.zeros_like(self.x)

		_x = self.x# - self.momentum*self.gradient

		for i, _ in enumerate(_x):
			# i 番目の変数で偏微分する
			_grad = self.numerical_diff(_x, i)

		self.h = self.alpha*self.h + self.alpha_*_grad*_grad
		self.gradient = self.momentum*self.gradient - self.learning_rate*_grad/(np.sqrt(self.h)+self.eps)
		# 更新
		self.next_pos = self.x + self.gradient

# AdaDelta
class AdaDeltaOptimizer(Optimizer):
	def __init__(self, f, init_pos, learning_rate=0.01, gamma=0.95, eps=1e-6, name=None, color="red"): # gammaはMomentumみたいなパラメータのやつのこと。
		super(AdaDeltaOptimizer, self).__init__(f, init_pos, learning_rate, name, color)
		self.h = np.zeros_like(init_pos)
		self.s = np.zeros_like(init_pos)
		self.eps = eps 						# 論文では1e-6推奨らしい。
		self.gamma = gamma 					# gammanのチェックは特にしません。
		self.gamma_ = 1.0-gamma

	def optimize(self):
		for i, _ in enumerate(self.x):
			# i 番目の変数で偏微分する
			self.gradient[i] = self.numerical_diff(self.x, i)

		self.h = self.gamma*self.h + self.gamma_*self.gradient*self.gradient
		v = np.sqrt(self.s+self.eps)/np.sqrt(self.h+self.eps)*self.gradient
		self.s = self.gamma*self.s + self.gamma_*v*v
		# 更新
		self.next_pos = self.x - v

# Adam
class AdamOptimizer(Optimizer):
	def __init__(self, f, init_pos, learning_rate=0.01, alpha=0.001, beta_1=0.9, beta_2=0.999, eps=1e-7, name=None, color="red"): # gammaはMomentumみたいなパラメータのやつのこと。
		super(AdamOptimizer, self).__init__(f, init_pos, learning_rate, name, color)
		self.m = np.zeros_like(init_pos)
		self.v = np.zeros_like(init_pos)
		self.eps = eps 						# 論文では1e-6推奨らしい。
		self.alpha = alpha
		self.beta_1 = beta_1
		self.beta_1_ = 1.0-beta_1
		self.beta_2 = beta_2
		self.beta_2_ = 1.0-beta_2
		self.iteration_beta_1 = beta_1
		self.iteration_beta_2 = beta_2

	def optimize(self):
		for i, _ in enumerate(self.x):
			# i 番目の変数で偏微分する
			self.gradient[i] = self.numerical_diff(self.x, i)

		self.m = self.beta_1*self.m + self.beta_1_*self.gradient
		self.v = self.beta_2*self.v + self.beta_2_*self.gradient*self.gradient

		_m = self.m / (1.0-self.iteration_beta_1)
		_v = self.v / (1.0-self.iteration_beta_2)

		# 更新
		self.next_pos = self.x - self.alpha*_m/(np.sqrt(_v)+self.eps)
		self.iteration_beta_1 *= self.beta_1
		self.iteration_beta_2 *= self.beta_2
