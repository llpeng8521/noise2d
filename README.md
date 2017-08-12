# 二维噪声的生成函数

## 说明
1. 使用python numpy生成的二维噪声信息
2. 提供一维噪声和二维噪声的生成算法
3. 生成噪声数据需要依赖numpy库
4. 测试噪声库需要依赖matplotlib库

## 参考文档
1. http://www.redblobgames.com/articles/noise/introduction.html
2. http://www.redblobgames.com/articles/noise/2d/

## 一维噪声函数说明

* make_noise(N, exp=0) 通用生成噪声的函数，默认白噪声
* red_noise(N) 生成红噪声函数 exp=-2
* pink_noise(N) 生成粉红噪声函数 exp=-1
* white_noise(N) 生成白噪声函数 exp=0
* blue_noise(N) 生成蓝噪声函数 exp=1
* violet_noise(N) 生成紫噪声函数 exp=2

## 二维噪声函数说明

* make_noise2d((M,N), exp=0, pxd=1) 通用生成噪声的函数，默认白噪声
* make_noise2d＿filter((M,N), cutin, cutoff, exp=0, pxd=1)  通用生成理想带通滤波后的噪声函数，默认白噪声
* red_noise2d((M,N)) 生成二维红噪声函数 exp=-2
* pink_noise2d((M,N)) 生成二维粉红噪声函数 exp=-1
* white_noise2d((M,N)) 生成二维白噪声函数 exp=0
* blue_noise2d((M,N)) 生成二维蓝噪声函数 exp=1
* violet_noise2d((M,N)) 生成二维紫噪声函数 exp=2