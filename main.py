# -*- coding: utf-8 -*-

"""
Author : mau
Tested in python2
Thank to the all samples in the Internet.
Following code is not checking arguments.
"""
from preset import Preset
from drawer import Drawer

"""
# 自分で設定するなら
import numpy as np
from optimizers import *
from graph import Graph
from functions import *
"""

def main():
	#　設定書くと見づらいからプリセットを使う
	preset = Preset()

	# このコメント書いてる時は1~9のプリセット
	optimizer_list, graph = preset.preset_5()

	# 描画のためのセットアップ
	drawer = Drawer(optimizer_list, graph)

	# アニメーション作成
	drawer.animation()

	# 保存
	#drawer.save_animation("optimization.gif", "./")
	# 表示
	drawer.show()

if __name__ == '__main__':
	main()