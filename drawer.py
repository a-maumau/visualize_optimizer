# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

# グラフの回転具合
rotating_per_step = 2
# 勾配方向の長さ調節
gradient_vec_len = 0.5

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
			
			# for plotting gradient
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