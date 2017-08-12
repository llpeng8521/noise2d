# -*- coding:utf-8-*-
# Created by Li Lepeng on 2017-08-12

"""噪声生成库函数"""

from noise import make_noise
from noise import make_noise2d
from noise import make_noise2d_filter

"""
def make_noise(size, exp=0, normal=True, seed=None):
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

"""
def make_noise2d(size, exp=0, pxd=1, normal=True, seed=None):
	通用噪声生成函数
	@size: 噪声大小 M*N
	@exp: 指数系数(噪声颜色)
	@pxd: the number of pixels per unit of frequency
	@normal: 是否归一化
	@seed: 随机种子
"""

"""
def make_noise2d_filter(size, cutin, cutoff, exp=0, pxd=1, normal=True, seed=None):
	通用噪声生成函数
	@size: 噪声大小 M*N
	@cutin, cutoff: 截止频率
	@pxd: the number of pixels per unit of frequency
	@exp: 指数系数(噪声颜色)
	@normal: 是否归一化
	@seed: 随机种子
"""

def red_noise(size, normal=True, seed=None):
	"""红噪声"""
	return make_noise(size, -2, normal, seed)

def pink_noise(size, normal=True, seed=None):
	"""粉红噪声"""
	return make_noise(size, -1, normal, seed)

def white_noise(size, normal=True, seed=None):
	"""白噪声"""
	import numpy as np
	return np.random.random(size)

def blue_noise(size, normal=True, seed=None):
	"""蓝噪声"""
	return make_noise(size, 1, normal, seed)

def violte_noise(size, normal=True, seed=None):
	"""紫噪声"""
	return make_noise(size, 2, normal, seed)


def red_noise2d(size, normal=True, seed=None):
	"""红噪声"""
	return make_noise2d(size, -2, normal, seed)

def pink_noise2d(size, normal=True, seed=None):
	"""粉红噪声"""
	return make_noise2d(size, -1, normal, seed)

def white_noise2d(size, normal=True, seed=None):
	"""白噪声"""
	import numpy as np
	return np.random.random(size)

def blue_noise2d(size, normal=True, seed=None):
	"""蓝噪声"""
	return make_noise2d(size, 1, normal, seed)

def violte_noise2d(size, normal=True, seed=None):
	"""紫噪声"""
	return make_noise2d(size, 2, normal, seed)