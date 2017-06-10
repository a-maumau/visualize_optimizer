# -*- coding: utf-8 -*-

"""
all-in-one version

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
delta = 1e-4
# 勾配方向の長さ調節
gradient_vec_len = 0.5
# グラフの回転具合
rotating_per_step = 2

# グラフ
class Graph:
	def __init__(self, range_x, range_y, func=None):
		self.f = func if func is not None else self.default_function #ソースいじらずに自分で設定したい人用に準備しときました。
		self.xr, self.yr = np.meshgrid(range_x, range_y) # センス的にはx1, x2に合わせるべきか。
		self.z = self.f([self.xr, self.yr]) 			 # ただここのzをfと書くと被るしで微妙なので...

	def mesh(self):
		return [self.xr, self.yr, self.z]

	def default_function(self, x):
		#とりあえずこれって感じ。
		#return 1/10.0*x[0] ** 2 + 2*x[1] ** 2

		# 以下適当にいくつか最適化のベンチマーク関数を列挙する。
		# 激しいのは割愛。
		"""
			ackley's function
			search range
				x0 = [-32.768, -32.768]
				x1 = [-32.768, -32.768]
			global min : f(0, 0) = 0
		"""
		# あんまり面白くない。
		#return 20 - 20*np.exp(-0.2*np.sqrt(0.5*(x[0]**2 + x[1]**2))) - np.exp(0.5*(np.cos(2*math.pi*x[0])+np.sin(2*math.pi*x[1]))) + math.e
		# 波々のところの調整版
		#return 20 - 20*np.exp(-0.2*np.sqrt(0.5*(x[0]**2 + x[1]**2))) - np.exp(0.5*(np.cos(0.5*math.pi*x[0])+np.sin(0.5*math.pi*x[1]))) + math.e

		"""
		Rosenbrock function 通称バナナ関数。
		search range
			x0 = [-5, 5]
			x1 = [-5, 5]
		global min : f(1,1) = 0
		"""
		# うん。って感じ
		#return 100*(x[1] - x[0]**2)**2 + (x[0]-1)**2

		"""
		Booth's function
		search range
			x0 = [-10, 10]
			x1 = [-10, 10]
		global min : f(1,3) = 0
		"""
		# あんまりぱっとしない
		return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

		"""
		Three-hump camel function
		search range
			x0 = [-5, 5]
			x1 = [-5, 5]
		global min : f(0,0) = 0
		"""
		# グラフ全体見た感じだとわかりづらいけど一応波打って(0,0)に向かっているようだ。
		#return 2*x[0]**2 - 1.05*x[0]**4 + x[0]**6/6 + x[0]*x[1] + x[1]**2

		"""
		McCormick function
		search range
			x0 = [-1.5, 4]
			x1 = [-3, 4]
		global min : f(-0.54791,-1.54719) = 0
		"""
		# そうだな。って感じ。
		#return np.sin(x[0] + x[1]) + (x[0] - x[1])**2 -1.5*x[0] + 2.5*x[1] + 1

		"""
		five-well potential function
		search range
			x0 = [-20, 20]
			x1 = [-20, 20]
		global min : f(4.92, -9:89) = -1.4616
		"""
		"""
		# 微妙に図と違うのが出来上がってるのはなぜなのかよく分からない。
		return (1.0 -1.0/(1.0+0.05*(x[0]**2 + (x[1]-10))**2)
					-1.0/(1.0+0.05*((x[0]-10)**2 + x[1]**2))
					-1.5/(1.0+0.03*((x[0]+10)**2 + x[1]**2))
					-2.0/(1.0+0.05*((x[0]-5)**2 + (x[1]+10)**2))
					-1.0/(1.0+0.1*((x[0]+5)**2 + (x[1]+10)**2))
				) * (1.0+0.0001*(x[0]**2 + x[1]**2)**1.2)
		"""

		
		# five-wellちょっといじったらいい感じのになったのでの残しておく。
		# 個人的には一番オススメ。絵に描いたような複数の局所解のあるグラフ
		"""
		return (1.0 -1.0/(1.0+0.05*(x[0]**2 + (x[1]-10)**2))
					-1.0/(1.0+0.05*((x[0]-10)**2 + x[1]**2))
					-1.5/(1.0+0.03*((x[0]+10)**2 + x[1]**2))
					-2.0/(1.0+0.05*((x[0]-5)**2 + (x[1]+10)**2))
					-1.0/(1.0+0.1*((x[0]+5)**2 + (x[1]+10)**2))
				) * (1.0+0.0001*(x[0]**2 + x[1]**2)**1.2) +1.5 #ここの1.5でちょっと調整
		"""

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
	def optimize(self):
		# 勾配を入れるベクトルをゼロで初期化する
		grad = np.zeros_like(self.x)

		for i, _ in enumerate(self.x):
			# i 番目の変数で偏微分する + ノイズを入れる。
			self.gradient[i] = self.numerical_diff(self.x, i)*(np.random.random()*4.0 - 0.5) + (np.random.random()*1 - 0.5)

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

# 描画用
class Drawer():
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
		self.ax3d.cla()
		self.ax2d.cla()

		# plot title
		#plt.title("step " + 'i=' + str(frame))
		self.fig.suptitle('calc. step = % 3d' % (step), fontsize=14, fontweight='bold')

		self.ax3d.view_init(40, rotating_per_step*step)

		self.ax3d.plot_surface(self.graph_mesh[0], self.graph_mesh[1], self.graph_mesh[2], alpha = 0.5, cmap=plt.cm.coolwarm)
		self.ax2d.contourf(self.graph_mesh[0], self.graph_mesh[1], self.graph_mesh[2], zdir='z', offset=-1, cmap=plt.cm.coolwarm)

		for index, optimizer in enumerate(self.optimizers):
			optimizer.update()

			pos = optimizer.pos()
			self.pos_array[index][0].append(pos[0])
			self.pos_array[index][1].append(pos[1])
			self.pos_array[index][2].append(pos[2])

			self.ax3d.plot([pos[0]], [pos[1]], [pos[2]], 'o', c=optimizer.color)
			self.ax3d.plot(self.pos_array[index][0], self.pos_array[index][1], self.pos_array[index][2], alpha = 0.8, c=optimizer.color, 
							label="{0:} ({1:>8.3f}, {2:>8.3f}, {3:>8.3f})".format(optimizer.name, pos[0], pos[1], pos[2]))
			
			# plot gradient
			#grad = optimizer.pos_gradient()
			#grad_norm = np.linalg.norm(grad)+1e-6
			#self.ax3d.plot([pos[0], pos[0]-grad[0]/grad_norm*gradient_vec_len], [pos[1], pos[1]-grad[1]/grad_norm*gradient_vec_len], zs=[pos[2], pos[2]], c='green')

			self.ax2d.plot([pos[0]], [pos[1]], 'o', c=optimizer.color)
			self.ax2d.plot(self.pos_array[index][0], self.pos_array[index][1], c=optimizer.color, alpha = 0.8)

		# 0.99系での応急処置
		plt.legend(loc="upper center", 
				   bbox_to_anchor=(-0.1,-0.08), # 描画領域の少し下にBboxを設定
				   ncol=2						# 2列
				  )

# 更新度合い　勾配具合違うのでうまく調整して
#learning_rate = 2.5 #five-wellでちょうどいい感じ
learning_rate = 0.005
momentum = 0.7

def main():
	# おまじない
	fig = plt.figure(figsize=(10, 6))
	ax2d = fig.add_subplot(1, 2, 1, adjustable='box', aspect=1)
	ax3d = fig.add_subplot(1, 2, 2, projection='3d', adjustable='box', aspect=1)

	# グラフと範囲 基本にはグラフの探索範囲に合わせると良い。
	x0 = np.arange(-5.0, 5.0, 0.25)
	x1 = np.arange(-5.0, 5.0, 0.25)
	G = Graph(x0, x1)

	# 初期値
	#init_pos = 10*np.random.random(2)-5
	init_pos = np.array([3.0,4.0])
	init_pos = np.array([-2.0,-4.0])
	#init_pos = np.array([6.0,8.0])
	# 最適化er
	gd_opt = GDOptimizer(G.f, init_pos, learning_rate, "GD", "red")
	sgd_opt = SGDOptimizer(G.f, init_pos, learning_rate, "SGD", "firebrick")
	mom_opt = MomentumOptimizer(G.f, init_pos, learning_rate, momentum=momentum, name="Mom", color="green")
	nag_opt = NAGOptimizer(G.f, init_pos, learning_rate, momentum, "NAG", "lime")
	ada_grad_opt = AdaGradOptimizer(G.f, init_pos, learning_rate, eps=1e-7, name="AdaGrad", color="yellow")
	rmsp_opt = RMSpropOptimizer(G.f, init_pos, learning_rate, alpha=0.9, name="RMSprop", color="blue")
	rmsp_mom_opt = RMSpropMomentumOptimizer(G.f, init_pos, learning_rate, alpha=0.2, momentum=momentum, name="RMSprop+Momentum", color="cyan")
	ada_del_opt = AdaDeltaOptimizer(G.f, init_pos, gamma=0.9, eps=1e-2, name="AdaDelta", color="purple")
	adam_opt = AdamOptimizer(G.f, init_pos, alpha=0.2, beta_1=0.8, beta_2=0.9, eps=1e-7, name="Adam", color="deeppink")

	# 描画のためのセットアップ
	plot = Drawer(fig, ax2d, ax3d, [gd_opt, sgd_opt, mom_opt, ada_grad_opt, rmsp_opt, rmsp_mom_opt, ada_del_opt, adam_opt], G)

	# アニメーション作成
	ani = animation.FuncAnimation(fig, plot.draw, frames = 300, interval = 50) 

	# 保存
	#ani.save("compute.gif", writer = 'imagemagick')
	# 表示
	plt.show()

if __name__ == '__main__':
	main()