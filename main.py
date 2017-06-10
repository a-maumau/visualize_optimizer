# -*- coding: utf-8 -*-

"""
Author : mau
Tested in python2
Thank to the all samples in the Internet.
Following code is not checking argments
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from preset import Preset
from drawer import Drawer

"""
# 自分で設定するなら
import numpy as np
from optimizers import *
from graph import Graph
from functions import *
"""

# 更新度合い　勾配具合違うのでうまく調整して
#learning_rate = 2.5 #five-wellでちょうどいい感じ
learning_rate = 0.1
momentum = 0.7

def main():
	# おまじない
	fig = plt.figure(figsize=(10, 6))
	# 左の等高グラフ
	ax2d = fig.add_subplot(1, 2, 1, adjustable='box', aspect=1)
	# 右の３次元グラフ
	ax3d = fig.add_subplot(1, 2, 2, projection='3d', adjustable='box', aspect=1)

	#　設定書くと見づらいからプリセットを使う
	preset = Preset()

	optimizer_list, graph = preset.preset_1()

	# 描画のためのセットアップ
	plot = Drawer(fig, ax2d, ax3d, optimizer_list, graph)

	# アニメーション作成
	ani = animation.FuncAnimation(fig, plot.draw, frames = 300, interval = 50) 

	# 保存
	#ani.save("compute.gif", writer = 'imagemagick')
	# 表示
	plt.show()

if __name__ == '__main__':
	main()