import cv2
import numpy
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

def 灰度图水印(图片:numpy.ndarray, 水印内容:str, 水印模式:str = "增亮"):
	# 图像基本信息
	图片高,   图片长   = 图片.shape
	水印中心y, 水印中心x = 图片高 // 4, 图片长 // 4  # 中心点坐标、
	# 2. 傅里叶变换
	傅里叶图像         = cv2.dft(numpy.float32(图片), flags=cv2.DFT_COMPLEX_OUTPUT)
	傅里叶图像_低频中心 = numpy.fft.fftshift(傅里叶图像)  # 低频移至中心

	# 3. 使用PIL创建频域蒙版
	# ---- 创建透明蒙版画布 ----
	PIL掩膜 = PIL.Image.new(
		mode  = "L",             # 灰度模式
		size  = (图片长, 图片高), # 参数顺序与opencv相反
		color = 0                # 全黑
	)
	PIL掩膜绘制对象 = PIL.ImageDraw.Draw(PIL掩膜)

	# ---- 设置字体参数 ----
	字体大小 = 50  # 可调整
	try:
		字体 = PIL.ImageFont.truetype(
			font     = r"C:/Windows/Fonts/simhei.ttf",
			size     = 字体大小,
			encoding = "utf-8"
		)  # 使用黑体
	except:
		字体 = PIL.ImageFont.load_default()  # 备用默认字体
	# ---- 计算文字居中位置 ----
	文字框         = PIL掩膜绘制对象.textbbox((0,0), 水印内容, font=字体)
	文字框长度     = 文字框[2] - 文字框[0]
	文字框高度     = 文字框[3] - 文字框[1]
	文字框起点位置 = (水印中心x - 文字框长度//2, 水印中心y - 文字框高度//2)  # 中心坐标
	# ---- 绘制半透明文字 ----
	PIL掩膜绘制对象.text(
		xy   = 文字框起点位置,
		text = 水印内容,
		fill = 255,
		font = 字体
	)  # fill为灰度值（0-255）
	# 转化为 numpy-OpenCV模式
	掩膜 = numpy.asarray(PIL掩膜).astype(numpy.float32)
	掩膜 += 掩膜[::-1,::-1] + 掩膜[:,::-1] + 掩膜[::-1,:]

	# 4. 将水印叠加到频域（增强幅度）
	print(numpy.max(傅里叶图像[掩膜 != 0]))
	if 水印模式 == "增亮":
		alpha = 0.001 * numpy.max(傅里叶图像[掩膜 != 0])  # 水印强度因子（动态选取傅里叶图像水印处最大像素点的0.01倍）
		新傅里叶图像_低频中心 = 傅里叶图像_低频中心 + alpha * cv2.merge([掩膜,掩膜])  # 直接修改复数域的实部和虚部
	else:
		新傅里叶图像_低频中心 = 傅里叶图像_低频中心 * (cv2.merge([掩膜,掩膜]) == 0)  # 直接修改复数域的实部和虚部

	cv2.imshow("new fft 0", cv2.resize(新傅里叶图像_低频中心[:,:,0],dsize=None, fx=0.5, fy=0.5))
	cv2.imshow("new fft 1", cv2.resize(新傅里叶图像_低频中心[:,:,1],dsize=None, fx=0.5, fy=0.5))
	cv2.waitKey()
	cv2.destroyAllWindows()

	# 5. 逆傅里叶变换
	新傅里叶图像 = numpy.fft.ifftshift(新傅里叶图像_低频中心)
	新图像       = cv2.idft(新傅里叶图像)
	新图像       = cv2.magnitude(新图像[:, :, 0], 新图像[:, :, 1])

	# 6. 归一化并返回结果
	cv2.normalize(新图像, 新图像, 0, 255, cv2.NORM_MINMAX)
	return 新图像

def 傅里叶水印(输入图片路径, 输出图片路径):
	# 1. 读取图像并转为灰度
	输入图片 = cv2.imread(输入图片路径, cv2.IMREAD_COLOR)
	
	水印模式 = "增亮" # 参数范围：增亮, 减暗
	if (输入图片路径.endswith(".jpg")):
		水印模式 = "增亮" # jpg 往往是照片，目前发现增亮型水印不会造成太大影响，但需要进一步实验
	elif (输入图片路径.endswith(".png")):
		水印模式 = "减暗" # png 往往是画作，目前发现增亮型水印会造成图片整体亮度减小，但需要进一步实验
	else:
		# 其他图片模式暂且不清楚，具体问题具体分析
		pass
	print(水印模式)
	蓝色通道, 绿色通道, 红色通道 = cv2.split(输入图片)
	蓝色新图像 = 灰度图水印(蓝色通道, "from: 水印测试",   水印模式)
	绿色新图像 = 灰度图水印(绿色通道, "to:   水印测试",   水印模式)
	红色新图像 = 灰度图水印(红色通道, "content:水印测试", 水印模式)
	新图像 = cv2.merge([蓝色新图像, 绿色新图像, 红色新图像])

	cv2.imwrite(输出图片路径, numpy.uint8(新图像))

# 调用函数
傅里叶水印("./input/sunset.jpg", "./output/sunset_watermarked.png")
傅里叶水印("./input/catgirl.png", "./output/catgirl_watermarked.png")