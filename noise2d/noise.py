# -*- coding:utf-8-*-
# Created by Li Lepeng on 2017-08-12

"""一维噪声生成函数"""

import numpy as np


def normalize(array):
	"""归一化函数"""
	minv = array.min()
	maxv = array.max()
	return (array - minv) / (maxv - minv)


def make_noise(size, exp=0, normal=True, seed=None):
	"""
	通用噪声生成函数
	@size: 噪声的长度
	@exp: 指数系数(颜色噪声) (可以取任意值，下面的值仅供参考)
		-2: rednoise
		-1: pinknoise
		0: whitenoise
		1: bluenoise
		2: violetnoise
	@normal: 是否归一化
	@seed: 随机种子
	"""

	# 1.产生随机噪声
	if seed:
		rndstate = np.random.RandomState(seed)
		signal =  rndstate.rand(size)
	else:
		signal = np.random.random(size)

	# 2.fft转换到频域
	signal_fft = np.fft.rfft(signal)

	# 3.频率幅值(权值)函数
	freq = np.linspace(1, size/2 + 1, size/2 + 1)**exp
	signal_filter = signal_fft * freq

	# 4.反变换到空域(时域)
	noise = np.fft.irfft(signal_filter)
	return normalize(noise) if normal else noise


def test_noise():
	"""测试噪声函数"""
	import matplotlib.pyplot as plt
	size = 50

	def plot(r, c, idx, y):
		x = range(0, len(y))
		plt.subplot(r, c, idx)
		y = normalize(y)
		plt.axis([0, len(y), -0.2, 1.2])
		plt.bar(x, y)

	seed = 1
	plot(5, 1, 1, make_noise(size, -2, seed=seed))
	plot(5, 1, 2, make_noise(size, -1, seed=seed))
	plot(5, 1, 3, make_noise(size, 0, seed=seed))
	plot(5, 1, 4, make_noise(size, 1, seed=seed))
	plot(5, 1, 5, make_noise(size, 2, seed=seed))
	plt.show()

# test_noise()