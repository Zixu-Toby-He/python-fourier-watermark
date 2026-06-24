import cv2
import numpy
import matplotlib.pyplot

def 灰度傅里叶解密(灰度图像, 显示标题:str = "title"):
	# 1. 傅里叶变换并中心化
	傅里叶图像          = cv2.dft(numpy.float32(灰度图像), flags=cv2.DFT_COMPLEX_OUTPUT)
	傅里叶图像_低频中心 = numpy.fft.fftshift(傅里叶图像)  # 低频移至中心
	半宽图片 = 傅里叶图像_低频中心[0:(傅里叶图像_低频中心.shape[0] // 2),0:(傅里叶图像_低频中心.shape[1] // 2)]

	# 3. 计算幅度谱（增强显示）
	增强图像 = cv2.magnitude(半宽图片[:, :, 0], 半宽图片[:, :, 1])
	对数增强图像 = (20 * numpy.log(增强图像 + 1))**0.3  # 对数变换增强细节

	# 3. 可视化频域水印
	matplotlib.pyplot.figure(figsize=(6, 6))
	matplotlib.pyplot.imshow(对数增强图像, cmap='gray')
	matplotlib.pyplot.title(显示标题)
	matplotlib.pyplot.axis('off')
	对数 = numpy.zeros_like(对数增强图像)
	对数 = cv2.normalize(
		src=对数增强图像,
		dst=对数,
		alpha=0,
		beta=255,
		norm_type=cv2.NORM_MINMAX
	).astype(numpy.uint8)
	cv2.imshow(显示标题, 对数)
	cv2.waitKey(1)
	matplotlib.pyplot.show()
	cv2.destroyAllWindows()

def 傅里叶解密(输入图片路径):
	输入图片 = cv2.imread(输入图片路径, cv2.IMREAD_COLOR)
	蓝色通道, 绿色通道, 红色通道 = cv2.split(输入图片)
	蓝色新图像 = 灰度傅里叶解密(蓝色通道, "blue code")
	绿色新图像 = 灰度傅里叶解密(绿色通道, "green code")
	红色新图像 = 灰度傅里叶解密(红色通道, "red code")

傅里叶解密("./output/sunset_watermarked.png")
傅里叶解密("./output/catgirl_watermarked.png")
