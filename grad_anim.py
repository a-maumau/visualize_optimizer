# -*- coding: utf-8 -*-

"""
Author : mau
Tested in python2
Thank to the all samples in the Internet.
Following code is not checking argments
"""

import abc
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# 数値微分の微小区間値
delta = 1e-5
# 勾配方向の長さ調節
gradient_vec_len = 1.0
# 更新度合い
learning_rate = 0.5
# グラフの回転具合
rotating_per_step = 2

# グラフ
class Graph:
	def __init__(self, range_x, range_y):
		self.xr, self.yr = np.meshgrid(range_x, range_y) # センス的にはx1, x2に合わせるべきか。
		self.z = self.f([self.xr, self.yr]) 			 # ただここのzをfと書くと被るしで微妙なので...

	def f(self, x):
		return 2*x[0] ** 2 + 0.1*x[1] ** 2

	def mesh(self):
		return [self.xr, self.yr, self.z]

# 最適化手法
class Optimizer(object):
	__metaclass__ = abc.ABCMeta

	def __init__(self, f, init_pos, learning_rate, name=None, color="red"):
		self.f = f
		self.learning_rate = learning_rate
		self.x = init_pos
		self.next_pos = init_pos
		self.gradient = np.zeros_like(init_pos)

		self.name = name
		self.color = color

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
	def numerical_diff(self, x, i):
		"""
		#中央差分・数値微分
		#i : 偏微分する変数のインデックス
		"""
		h_vec = np.zeros_like(x)
		h_vec[i] = delta

		# 数値微分を使って偏微分する
		return (self.f(x + h_vec) - self.f(x - h_vec)) / (2 * delta)

	def optimize(self):
		# 勾配を入れるベクトルをゼロで初期化する
		grad = np.zeros_like(self.x)

		for i, _ in enumerate(self.x):
			# i 番目の変数で偏微分する
			grad[i] = self.numerical_diff(self.x, i)

		self.gradient = grad
		# 更新
		self.next_pos = self.x -grad*self.learning_rate

# 描画用
class Drawing():
	def __init__(self, fig, ax2d, ax3d, optimizers, graph):
		self.graph_mesh = graph.mesh()
		self.optimizers = optimizers
		# 失敗例
		#self.pos_array = [[[],[],[]]] * len(optimizers)
		self.pos_array = [ [[],[],[]] for i in range(len(optimizers)) ]

		self.fig = fig
		self.ax2d = ax2d
		self.ax3d = ax3d

	def draw(self, step):
		# clear buffer
		ax3d.cla()
		ax2d.cla()

		# plot title
		#plt.title("step " + 'i=' + str(frame))
		fig.suptitle('calc. step = % 3d' % (step), fontsize=14, fontweight='bold')

		self.ax3d.view_init(40, rotating_per_step*step)

		self.ax3d.plot_surface(self.graph_mesh[0], self.graph_mesh[1], self.graph_mesh[2], alpha = 0.5, cmap=cm.coolwarm)
		self.ax2d.contourf(self.graph_mesh[0], self.graph_mesh[1], self.graph_mesh[2], zdir='z', offset=-1, cmap=plt.cm.coolwarm)

		for index, optimizer in enumerate(self.optimizers):
			optimizer.update()

			pos = optimizer.pos()
			self.pos_array[index][0].append(pos[0])
			self.pos_array[index][1].append(pos[1])
			self.pos_array[index][2].append(pos[2])

			self.ax3d.plot([pos[0]], [pos[1]], [pos[2]], 'o', c=optimizer.color)
			self.ax3d.plot(self.pos_array[index][0], self.pos_array[index][1], self.pos_array[index][2], label=optimizer.name, c=optimizer.color)
			
			# plot gradient
			#grad = optimizer.pos_gradient()
			#grad_norm = np.linalg.norm(grad)+1e-6
			#self.ax3d.plot([pos[0], pos[0]-grad[0]/grad_norm*gradient_vec_len], [pos[1], pos[1]-grad[1]/grad_norm*gradient_vec_len], zs=[pos[2], pos[2]], c='green')

			self.ax2d.plot([pos[0]], [pos[1]], 'o', c=optimizer.color)
			self.ax2d.plot(self.pos_array[index][0], self.pos_array[index][1], c=optimizer.color)

		# 0.99系での応急処置
		plt.legend(loc="upper center", 
				   bbox_to_anchor=(0.5,-0.05), # 描画領域の少し下にBboxを設定
				   ncol=2						# 2列
				  )

def main():
	# おまじない
	fig = plt.figure(figsize=(10, 5))
	ax2d = fig.add_subplot(1, 2, 1, adjustable='box', aspect=1)
	ax3d = fig.add_subplot(1, 2, 2, projection='3d', adjustable='box', aspect=1)

	# グラフと範囲
	x0 = np.arange(-10.0, 10.0, 0.25)
	x1 = np.arange(-10.0, 10.0, 0.25)
	G = Graph(x0, x1)

	# 初期値
	init_pos = np.array([5.0,9.0])
	# 最適化er
	gd_opt = GDOptimizer(G.f, init_pos, learning_rate, "GD", "red")

	# 初期値
	init_pos = np.array([-9.0,-4.0])
	# 最適化er2
	gd_opt2 = GDOptimizer(G.f, init_pos, learning_rate, "GD2", "blue")

	# 描画のためのセットアップ
	plot = Drawing(fig, ax2d, ax3d, [gd_opt, gd_opt2], G)

	# アニメーション作成
	ani = animation.FuncAnimation(fig, plot.draw, frames = 300, interval = 100) 

	# 保存
	#ani.save("compute.gif", writer = 'imagemagick')
	# 表示
	plt.show()

if __name__ == '__main__':
	main()