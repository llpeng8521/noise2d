# -*- coding:utf-8-*-
# Created by Li Lepeng on 2017-08-12

"""二维噪声生成函数"""

import numpy as np


def normalize(array):
	"""归一化函数"""
	minv = array.min()
	maxv = array.max()
	return (array - minv) / (maxv - minv)


def gen_freq_func(size, exp=0, pxd=1):
	"""
	频域幅度(权值)函数
	@size: 噪声大小
	@exp: 指数系数(噪声颜色)
	@pxd: the number of pixels per unit of frequency
	"""
	pxd = float(pxd)
	rows, cols = size
	x = np.linspace(-0.5, 0.5, cols) * cols / pxd
	y = np.linspace(-0.5, 0.5, rows) * rows / pxd
	radius = np.sqrt((x**2)[np.newaxis] + (y**2)[:, np.newaxis])
	return radius ** exp


def make_noise2d(size, exp=0, pxd=1, normal=True, seed=None):
	"""
	通用噪声生成函数
	@size: 噪声大小 M*N
	@exp: 指数系数(噪声颜色)
	@normal: 是否归一化
	@seed: 随机种子
	"""

	# 1.产生随机噪声
	M, N = size
	if seed:
		rndstate = np.random.RandomState(seed)
		signal =  rndstate.rand(M, N)
	else:
		signal = np.random.random((M, N))

	# 2.fft转换到频域
	signal_fft2 = np.fft.fft2(signal)

	# 3.频率幅值(权值)函数
	freq_func = gen_freq_func((M, N), exp, pxd)
	signal_filter = np.fft.fftshift(signal_fft2) * freq_func

	# 4.反变换到空域(时域)
	noise = np.abs(np.fft.ifft2(np.fft.ifftshift(signal_filter)))
	return normalize(noise) if normal else noise


def gen_freq_filter_func(size, cutin, cutoff, exp=0, pxd=1):
	"""
	带通滤波处理的频域幅值(权值)函数
	@size: 噪声大小
	@cutin, cutoff: 截止频率
	@exp: 指数系数(噪声颜色)
	@pxd: the number of pixels per unit of frequency
	"""
	pxd = float(pxd)
	rows, cols = size
	x = np.linspace(-0.5, 0.5, cols) * cols / pxd
	y = np.linspace(-0.5, 0.5, rows) * rows / pxd
	radius = np.sqrt((x**2)[np.newaxis] + (y**2)[:, np.newaxis])
	filt = np.ones(size)
	for r in xrange(rows):
		for c in xrange(cols):
			if cutin <= radius[r][c] <= cutoff:
				filt[r][c] = radius[r][c] ** exp
			else:
				filt[r][c] = 0
	return filt


def make_noise2d_filter(size, cutin, cutoff, exp=0, pxd=1, normal=True, seed=None):
	"""
	通用噪声生成函数
	@size: 噪声大小 M*N
	@cutin, cutoff: 截止频率
	@exp: 指数系数(噪声颜色)
	@normal: 是否归一化
	@seed: 随机种子
	"""

	# 1.产生随机噪声
	M, N = size
	if seed:
		rndstate = np.random.RandomState(seed)
		signal =  rndstate.rand(M, N)
	else:
		signal = np.random.random((M, N))

	# 2.fft转换到频域
	signal_fft2 = np.fft.fft2(signal)

	# 3.频率幅值(权值)函数
	freq_func = gen_freq_filter_func((M, N), cutin, cutoff, exp, pxd)
	signal_filter = np.fft.fftshift(signal_fft2) * freq_func

	# 4.反变换到空域(时域)
	noise = np.abs(np.fft.ifft2(np.fft.ifftshift(signal_filter)))
	return normalize(noise) if normal else noise


def test_noise():
	"""测试噪声函数"""
	import matplotlib.pyplot as plt
	def plot(r, c, idx, y, title=""):
		plt.subplot(r, c, idx)
		plt.imshow(y, cmap="gray")
		plt.title(title)
		plt.axis("off")

	M, N = 300, 300
	# 1.原始噪声
	plt.figure(1)
	ori_noise = np.random.random((M, N))  # 产生白噪声
	plot(1, 2, 1, ori_noise, "Original Noise")

	ori_noise_fft = np.fft.fft2(ori_noise)
	ori_noise_fffshift = np.fft.fftshift(ori_noise_fft)
	plot(1, 2, 2, np.log(np.abs(ori_noise_fffshift)), "Original Noise FFT")

	plt.figure(2)
	# 2.红噪声exp=-2
	freqfilter = gen_freq_func((M, N), -2)
	noise_filter = ori_noise_fffshift * freqfilter
	noise = np.abs(np.fft.ifft2(np.fft.ifftshift(noise_filter)))
	plot(5, 3, 1, freqfilter, "Freq Func:exp=-2")
	plot(5, 3, 2, np.log(np.abs(noise_filter)), "Red Noise FFT")
	plot(5, 3, 3, noise, "Red Noise")

	# 3.粉红噪声exp=-1
	freqfilter = gen_freq_func((M, N), -1)
	noise_filter = ori_noise_fffshift * freqfilter
	noise = np.abs(np.fft.ifft2(np.fft.ifftshift(noise_filter)))
	plot(5, 3, 4, freqfilter, "Freq Func:exp=-1")
	plot(5, 3, 5, np.log(np.abs(noise_filter)), "Pink Noise FFT")
	plot(5, 3, 6, noise, "Pink Noise")

	# 4.白噪声exp=0
	freqfilter = gen_freq_func((M, N), 0)
	noise_filter = ori_noise_fffshift * freqfilter
	noise = np.abs(np.fft.ifft2(np.fft.ifftshift(noise_filter)))
	plot(5, 3, 7, freqfilter, "Freq Func:exp=0")
	plot(5, 3, 8, np.log(np.abs(noise_filter)), "White Noise FFT")
	plot(5, 3, 9, noise, "White Noise")

	# 5.蓝噪声exp=1
	freqfilter = gen_freq_func((M, N), 1)
	noise_filter = ori_noise_fffshift * freqfilter
	noise = np.abs(np.fft.ifft2(np.fft.ifftshift(noise_filter)))
	plot(5, 3, 10, freqfilter, "Freq Func:exp=1")
	plot(5, 3, 11, np.log(np.abs(noise_filter)), "Blue Noise FFT")
	plot(5, 3, 12, noise, "Blue Noise")

	# 6.紫噪声exp=2
	freqfilter = gen_freq_func((M, N), 2)
	noise_filter = ori_noise_fffshift * freqfilter
	noise = np.abs(np.fft.ifft2(np.fft.ifftshift(noise_filter)))
	plot(5, 3, 13, freqfilter, "Freq Func:exp=2")
	plot(5, 3, 14, np.log(np.abs(noise_filter)), "Violet Noise FFT")
	plot(5, 3, 15, noise, "Violet Noise")

	plt.show()


def test_noise_filter():
	"""测试带通滤波后的噪声函数"""
	import matplotlib.pyplot as plt
	def plot(r, c, idx, y, title=""):
		plt.subplot(r, c, idx)
		plt.imshow(y, cmap="gray")
		plt.title(title)
		plt.axis("off")

	M, N = 300, 300
	# 1.原始噪声
	plt.figure(1)
	ori_noise = np.random.random((M, N))  # 产生白噪声
	plot(1, 2, 1, ori_noise, "Original Noise")

	ori_noise_fft = np.fft.fft2(ori_noise)
	ori_noise_fffshift = np.fft.fftshift(ori_noise_fft)
	plot(1, 2, 2, np.log(np.abs(ori_noise_fffshift)), "Original Noise FFT")

	exp = 1

	plt.figure(2)
	# 2.颜色噪声(0~10)
	freqfilter = gen_freq_filter_func((M, N), 0, 10, exp)
	noise_filter = ori_noise_fffshift * freqfilter
	noise = np.abs(np.fft.ifft2(np.fft.ifftshift(noise_filter)))
	plot(4, 3, 1, freqfilter, "Freq Filter:(0, 10)")
	plot(4, 3, 2, np.abs(noise_filter), "Color Noise FFT")
	plot(4, 3, 3, noise, "Color Noise(0, 10)")

	# 3.颜色噪声(10~30)
	freqfilter = gen_freq_filter_func((M, N), 10, 30, exp)
	noise_filter = ori_noise_fffshift * freqfilter
	noise = np.abs(np.fft.ifft2(np.fft.ifftshift(noise_filter)))
	plot(4, 3, 4, freqfilter, "Freq Filter:(10, 30)")
	plot(4, 3, 5, np.abs(noise_filter), "Color Noise FFT")
	plot(4, 3, 6, noise, "Color Noise(10, 30)")

	# 4.颜色噪声(20~60)
	freqfilter = gen_freq_filter_func((M, N), 20, 60, exp)
	noise_filter = ori_noise_fffshift * freqfilter
	noise = np.abs(np.fft.ifft2(np.fft.ifftshift(noise_filter)))
	plot(4, 3, 7, freqfilter, "Freq Filter:(20, 60)")
	plot(4, 3, 8, np.abs(noise_filter), "Color Noise FFT")
	plot(4, 3, 9, noise, "Color Noise(20, 60)")

	# 5.颜色噪声(60~150)
	freqfilter = gen_freq_filter_func((M, N), 60, 150, exp)
	noise_filter = ori_noise_fffshift * freqfilter
	noise = np.abs(np.fft.ifft2(np.fft.ifftshift(noise_filter)))
	plot(4, 3, 10, freqfilter, "Freq Filter:(60, 150)")
	plot(4, 3, 11, np.abs(noise_filter), "Color Noise FFT")
	plot(4, 3, 12, noise, "Color Noise(60, 150)")

	plt.show()

# test_noise()
# test_noise_filter()
