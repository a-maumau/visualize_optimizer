# -*- coding: utf-8 -*-

import numpy as np
from optimizers import *
from graph import Graph
from functions import *

class Preset():
	# 比の違う２次関数
	def preset_1(self):
		# hyper parameter
		learning_rate = 0.1
		momentum = 0.9

		# グラフと範囲 基本にはグラフの探索範囲に合わせると良い。
		x0 = np.arange(-10.0, 10.0, 0.25)
		x1 = np.arange(-10.0, 10.0, 0.25)
		G = Graph(x0, x1)

		# 初期値
		#init_pos = 10*np.random.random(2)-5
		init_pos = np.array([-8.0,-4.0])
		# 最適化er
		gd_opt = GDOptimizer(G.f, init_pos, learning_rate, "GD", "red")
		#sgd_opt = SGDOptimizer(G.f, init_pos, learning_rate, name="SGD", color="firebrick")
		mom_opt = MomentumOptimizer(G.f, init_pos, learning_rate, momentum=momentum, name="Mom", color="green")
		nag_opt = NAGOptimizer(G.f, init_pos, learning_rate, momentum, "NAG", "lime")
		ada_grad_opt = AdaGradOptimizer(G.f, init_pos, learning_rate, eps=1e-7, name="AdaGrad", color="yellow")
		rmsp_opt = RMSpropOptimizer(G.f, init_pos, learning_rate, alpha=0.9, name="RMSprop", color="blue")
		rmsp_mom_opt = RMSpropMomentumOptimizer(G.f, init_pos, learning_rate, alpha=0.2, momentum=momentum, name="RMSprop+Momentum", color="cyan")
		ada_del_opt = AdaDeltaOptimizer(G.f, init_pos, gamma=0.9, eps=1e-2, name="AdaDelta", color="purple")
		adam_opt = AdamOptimizer(G.f, init_pos, alpha=0.2, beta_1=0.8, beta_2=0.9, eps=1e-7, name="Adam", color="deeppink")

		return [gd_opt, mom_opt, nag_opt, ada_grad_opt, rmsp_opt, rmsp_mom_opt, ada_del_opt, adam_opt], G

	def preset_2(self):
		learning_rate = 2.0 #five-wellでちょうどいい感じ
		momentum = 0.7
		
		x0 = np.arange(-20.0, 20.0, 0.25)
		x1 = np.arange(-20.0, 20.0, 0.25)
		f = FiveWellPotentialFunction_mod()
		G = Graph(x0, x1, f())

		# 初期値
		init_pos = np.array([0.0, -1.0])
		# 最適化er
		gd_opt = GDOptimizer(G.f, init_pos, learning_rate, "GD", "red")
		#sgd_opt = SGDOptimizer(G.f, init_pos, learning_rate,
		#						noize_vec_mul=10.0, noize_vec_negbias=0.5, noize_const_mul=2.0, noize_const_negbias=1.0, name="SGD", color="firebrick")
		mom_opt = MomentumOptimizer(G.f, init_pos, learning_rate, momentum=momentum, name="Mom", color="green")
		nag_opt = NAGOptimizer(G.f, init_pos, learning_rate, momentum, "NAG", "lime")
		ada_grad_opt = AdaGradOptimizer(G.f, init_pos, learning_rate, eps=1e-7, name="AdaGrad", color="yellow")
		rmsp_opt = RMSpropOptimizer(G.f, init_pos, learning_rate, alpha=0.8, name="RMSprop", color="blue")
		rmsp_mom_opt = RMSpropMomentumOptimizer(G.f, init_pos, learning_rate, alpha=0.2, momentum=momentum, name="RMSprop+Momentum", color="cyan")
		ada_del_opt = AdaDeltaOptimizer(G.f, init_pos, gamma=0.9, eps=1e-2, name="AdaDelta", color="purple")
		adam_opt = AdamOptimizer(G.f, init_pos, alpha=0.2, beta_1=0.8, beta_2=0.9, eps=1e-7, name="Adam", color="deeppink")

		return [gd_opt, mom_opt, nag_opt, ada_grad_opt, rmsp_opt, rmsp_mom_opt, ada_del_opt, adam_opt], G

	# Booth関数の上でいろいろ
	def preset_3(self):
		learning_rate = 0.05
		momentum = 0.7

		x0 = np.arange(-5.0, 5.0, 0.25)
		x1 = np.arange(-5.0, 5.0, 0.25)
		f = BoothFunction()
		G = Graph(x0, x1, f())

		# 初期値
		init_pos = np.array([-2.0,-4.0])
		# 最適化er
		gd_opt = GDOptimizer(G.f, init_pos, learning_rate, "GD", "red")
		mom_opt = MomentumOptimizer(G.f, init_pos, learning_rate, momentum=momentum, name="Mom", color="green")
		nag_opt = NAGOptimizer(G.f, init_pos, learning_rate, momentum, "NAG", "lime")
		ada_grad_opt = AdaGradOptimizer(G.f, init_pos, learning_rate, eps=1e-7, name="AdaGrad", color="yellow")
		rmsp_opt = RMSpropOptimizer(G.f, init_pos, learning_rate, alpha=0.9, name="RMSprop", color="blue")
		rmsp_mom_opt = RMSpropMomentumOptimizer(G.f, init_pos, learning_rate, alpha=0.2, momentum=momentum, name="RMSprop+Momentum", color="cyan")
		ada_del_opt = AdaDeltaOptimizer(G.f, init_pos, gamma=0.9, eps=1e-2, name="AdaDelta", color="purple")
		adam_opt = AdamOptimizer(G.f, init_pos, alpha=0.2, beta_1=0.8, beta_2=0.9, eps=1e-7, name="Adam", color="deeppink")

		return [gd_opt, mom_opt, nag_opt, ada_grad_opt, rmsp_opt, rmsp_mom_opt, ada_del_opt, adam_opt], G

	# 確率的（ノイズというなの乱数入れてるだけの実装）な勾配法のイメージと比較
	def preset_4(self):
		learning_rate = 3.0
		momentum = 0.7
		np.random.seed(7)

		x0 = np.arange(-20.0, 20.0, 0.25)
		x1 = np.arange(-20.0, 20.0, 0.25)
		f = FiveWellPotentialFunction_mod()
		G = Graph(x0, x1, f())

		# 初期値
		init_pos = np.array([-4.0,-4.0])
		# 最適化er
		gd_opt = GDOptimizer(G.f, init_pos, learning_rate, "GD", "red")
		mom_opt = MomentumOptimizer(G.f, init_pos, learning_rate, momentum=momentum, name="Mom", color="green")

		sgd_opt = SGDOptimizer(G.f, init_pos, learning_rate,
								noize_vec_mul=10.0, noize_vec_negbias=0.5, noize_const_mul=2.0, noize_const_negbias=1.0, name="SGD", color="blue")

		return [gd_opt, sgd_opt, mom_opt], G

	# MomentumとNAGの比較用。
	def preset_5(self):
		learning_rate = 0.02
		momentum = 0.9
		
		x0 = np.arange(-10.0, 10.0, 0.25)
		x1 = np.arange(-10.0, 10.0, 0.25)
		G = Graph(x0, x1)

		# 初期値
		init_pos = np.array([-9.0,-7.0])
		# 最適化er
		mom_opt = MomentumOptimizer(G.f, init_pos, learning_rate, momentum=momentum, name="Mom", color="green")
		nag_opt = NAGOptimizer(G.f, init_pos, learning_rate, momentum, name="NAG", color="lime")

		return [mom_opt, nag_opt], G

	# RMSpropとRMSporp+Momentum
	def preset_6(self):
		learning_rate = 0.02
		momentum = 0.9
		
		x0 = np.arange(-10.0, 10.0, 0.25)
		x1 = np.arange(-10.0, 10.0, 0.25)
		G = Graph(x0, x1)

		# 初期値
		init_pos = np.array([-9.0,-7.0])
		# 最適化er
		rmsp_opt = RMSpropOptimizer(G.f, init_pos, learning_rate, name="RMSprop", color="blue")
		rmsp_mom_opt = RMSpropMomentumOptimizer(G.f, init_pos, learning_rate, momentum=momentum, name="RMSprop+Momentum", color="cyan")

		return [rmsp_opt, rmsp_mom_opt], G

	# Ada系
	def preset_7(self):
		learning_rate = 0.001
		momentum = 0.9
		
		x0 = np.arange(-10.0, 10.0, 0.25)
		x1 = np.arange(-10.0, 10.0, 0.25)
		G = Graph(x0, x1)

		# 初期値
		init_pos = np.array([-9.0,-7.0])
		# 最適化er
		ada_grad_opt = AdaGradOptimizer(G.f, init_pos, learning_rate, eps=1e-7, name="AdaGrad", color="yellow")
		ada_del_opt = AdaDeltaOptimizer(G.f, init_pos, gamma=0.9, eps=1e-2, name="AdaDelta", color="purple")
		adam_opt = AdamOptimizer(G.f, init_pos, alpha=0.2, beta_1=0.8, beta_2=0.9, eps=1e-7, name="Adam", color="deeppink")
		smorms_opt = SMORMS3Optimizer(G.f, init_pos, learning_rate, name="SMORMS3", color="orange")

		return [ada_grad_opt, ada_del_opt, adam_opt, smorms_opt], G

	# Ada系2
	def preset_8(self):
		learning_rate = 0.001
		momentum = 0.9
		
		x0 = np.arange(-5.0, 5.0, 0.25)
		x1 = np.arange(-5.0, 5.0, 0.25)
		f = BoothFunction()
		G = Graph(x0, x1, f())

		# 初期値
		init_pos = np.array([-2.0,-4.0])
		# 最適化er
		ada_grad_opt = AdaGradOptimizer(G.f, init_pos, learning_rate, eps=1e-7, name="AdaGrad", color="yellow")
		ada_del_opt = AdaDeltaOptimizer(G.f, init_pos, gamma=0.9, eps=1e-2, name="AdaDelta", color="purple")
		adam_opt = AdamOptimizer(G.f, init_pos, alpha=0.2, beta_1=0.8, beta_2=0.9, eps=1e-7, name="Adam", color="deeppink")
		smorms_opt = SMORMS3Optimizer(G.f, init_pos, learning_rate, name="SMORMS3", color="orange")

		return [ada_grad_opt, ada_del_opt, adam_opt, smorms_opt], G

	# Ada系3
	def preset_9(self):
		learning_rate = 0.001
		momentum = 0.9
		
		x0 = np.arange(-20.0, 20.0, 0.25)
		x1 = np.arange(-20.0, 20.0, 0.25)
		f = FiveWellPotentialFunction_mod()
		G = Graph(x0, x1, f())

		# 初期値
		init_pos = np.array([0.0, -1.0])
		# 最適化er
		ada_grad_opt = AdaGradOptimizer(G.f, init_pos, learning_rate, eps=1e-7, name="AdaGrad", color="yellow")
		ada_del_opt = AdaDeltaOptimizer(G.f, init_pos, gamma=0.9, eps=1e-2, name="AdaDelta", color="purple")
		adam_opt = AdamOptimizer(G.f, init_pos, alpha=0.2, beta_1=0.8, beta_2=0.9, eps=1e-7, name="Adam", color="deeppink")
		smorms_opt = SMORMS3Optimizer(G.f, init_pos, learning_rate, name="SMORMS3", color="orange")

		return [ada_grad_opt, ada_del_opt, adam_opt, smorms_opt], G