# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

# グラフの回転具合
rotating_per_step = 2
# 勾配方向の長さ調節
gradient_vec_len = 1.0

# 描画用
class Drawer():
	def __init__(self, optimizers, graph):
		# おまじない
		self.fig = plt.figure(figsize=(10, 6))
		# 左の等高グラフ
		self.ax2d = self.fig.add_subplot(1, 2, 1, adjustable='box', aspect=1)
		# 右の３次元グラフ
		self.ax3d = self.fig.add_subplot(1, 2, 2, projection='3d', adjustable='box', aspect=1)

		# アニメーション初期設定
		self.anime_frames = 300
		self.anime_interval = 100
		self.anime = None

		# メッシュの取得
		self.graph_mesh = graph.mesh()
		
		self.optimizers = optimizers

		# 失敗例
		#self.pos_array = [[[],[],[]]] * len(optimizers)
		self.pos_array = [ [[],[],[]] for i in range(len(optimizers)) ]

	def set_animation(self, frames, interval):
		"""
		frames : iteration数
		interval : 指定msごとにコールバックがかかる。書き出す場合は実質のfps
		実際には処理が重くて表示だけする際にはintervalの値に間に合ってない。
		"""
		self.frames = frames
		self.interval = interval

	def animation(self):
		self.anime = animation.FuncAnimation(self.fig, self.draw, frames=self.anime_frames, interval=self.anime_interval)

	def save_animation(self, save_name="animation.gif", save_dir="./"):
		self.anime.save(save_dir+save_name, writer='imagemagick')

	def show(self):
		plt.show()

	def draw(self, step):
		# clear buffer
		self.ax3d.cla()
		self.ax2d.cla()

		# plot title
		#plt.title("step " + 'i=' + str(frame))
		self.fig.suptitle('calc. step = % 3d' % (step), fontsize=14, fontweight='bold')

		self.ax3d.view_init(40, rotating_per_step*step)

		# グラフと等高図の描画
		self.ax3d.plot_surface(self.graph_mesh[0], self.graph_mesh[1], self.graph_mesh[2], alpha = 0.5, cmap=plt.cm.coolwarm)
		self.ax2d.contourf(self.graph_mesh[0], self.graph_mesh[1], self.graph_mesh[2], zdir='z', offset=-1, cmap=plt.cm.coolwarm)

		for index, optimizer in enumerate(self.optimizers):
			optimizer.update()

			# 移動経路用に位置を保存
			pos = optimizer.pos()
			self.pos_array[index][0].append(pos[0])
			self.pos_array[index][1].append(pos[1])
			self.pos_array[index][2].append(pos[2])

			# 3d plot
			self.ax3d.plot([pos[0]], [pos[1]], [pos[2]], 'o', c=optimizer.color)
			self.ax3d.plot(self.pos_array[index][0], self.pos_array[index][1], self.pos_array[index][2], alpha = 0.8, c=optimizer.color, 
							label="{0:} ({1:>8.3f}, {2:>8.3f}, {3:>8.3f})".format(optimizer.name, pos[0], pos[1], pos[2]))
			
			# 2d plot
			self.ax2d.plot([pos[0]], [pos[1]], 'o', c=optimizer.color)
			self.ax2d.plot(self.pos_array[index][0], self.pos_array[index][1], c=optimizer.color, alpha = 0.8)

			# for plotting gradient
			grad = optimizer.pos_gradient()
			grad_norm = np.linalg.norm(grad)+1e-6
			#self.ax3d.plot([pos[0], pos[0]-grad[0]/grad_norm*gradient_vec_len], [pos[1], pos[1]-grad[1]/grad_norm*gradient_vec_len], zs=[pos[2], pos[2]], c='green')
			self.ax2d.plot([pos[0], pos[0]-grad[0]/grad_norm*gradient_vec_len], [pos[1], pos[1]-grad[1]/grad_norm*gradient_vec_len], c='black')

		# 凡例用
		# 0.99系での応急処置
		plt.legend(loc="upper center", 
				   bbox_to_anchor=(-0.1,-0.08), # 描画領域の少し下にBboxを設定
				   ncol=2						# 2列
				  )
