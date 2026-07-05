import os
import ujson
import aicube
from libs.PipeLine import ScopedTiming
from libs.Utils import *
from media.sensor import *
from media.display import *
from media.media import *
import nncase_runtime as nn
import ulab.numpy as np  # 使用ulab.numpy适合嵌入式环境
import image
import gc
from machine import FPIOA, UART

# 预测框稳定器类
class BBoxStabilizer:
    def __init__(self, window_size=5, std_threshold=1.8, speed_threshold=40,
                 img_width=800, img_height=480, min_valid_area=300, edge_threshold=50):
        """
        初始化预测框稳定器（增强错误检测版）
        :param window_size: 滑动窗口大小（历史框数量），必须≥3
        :param std_threshold: 标准差阈值，超过此值视为异常波动，必须>0
        :param speed_threshold: 速度阈值，超过此值视为快速移动，必须≥0
        :param img_width: 图像宽度（固定为800），用于坐标范围校验
        :param img_height: 图像高度（固定为480），用于坐标范围校验
        :param min_valid_area: 最小有效框面积（像素²），过滤过小的错误框
        :param edge_threshold: 边缘阈值，距离图像边缘小于此值的框视为可疑边缘框
        """
        # 1. 参数合法性校验（防止无效配置）
        if window_size < 3:
            raise ValueError("窗口大小window_size必须≥3，否则无法计算有效的均值和标准差")
        if std_threshold <= 0:
            raise ValueError("标准差阈值std_threshold必须>0")
        if speed_threshold < 0:
            raise ValueError("速度阈值speed_threshold不能为负数")
        if img_width <= 0 or img_height <= 0:
            raise ValueError("图像宽高必须为正数（当前配置为800x480）")

        # 2. 核心参数初始化
        self.window_size = window_size
        self.std_threshold = std_threshold
        self.speed_threshold = speed_threshold

        # 3. 图像约束参数（针对480x800像素场景）
        self.img_width = img_width      # 800
        self.img_height = img_height    # 480
        self.min_valid_area = min_valid_area  # 过滤极小框（如1x1的错误框）
        self.edge_threshold = edge_threshold  # 边缘检测阈值（如左上角边缘）

        # 4. 滑动窗口及状态变量
        self.window = []                # 存储有效历史框
        self.prev_center = None         # 上一帧中心坐标
        self.suspicious_count = 0       # 连续可疑框计数（用于强化过滤）
        self.MAX_SUSPICIOUS = 2         # 最大连续可疑次数，超过则强制重置窗口

    def _is_valid_bbox(self, x1, y1, x2, y2):
        """判断当前框是否为有效框（过滤明显错误的框）"""
        # 1. 坐标范围校验（必须在图像内部）
        if x1 < 0 or y1 < 0 or x2 > self.img_width or y2 > self.img_height:
            return False

        # 2. 框完整性校验（x1 < x2，y1 < y2，避免无效框）
        if x1 >= x2 or y1 >= y2:
            return False

        # 3. 面积校验（过滤过小的错误框，如左上角1x1的噪声）
        area = (x2 - x1) * (y2 - y1)
        if area < self.min_valid_area:
            return False

        # 4. 边缘可疑框检测（左上角等边缘区域的框需额外警惕）
        is_edge_suspicious = (x1 < self.edge_threshold and y1 < self.edge_threshold)
        return not is_edge_suspicious  # 边缘可疑框直接判定为无效

    def _calc_center(self, x1, y1, x2, y2):
        """计算预测框中心坐标"""
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _calc_movement_speed(self, current_center):
        """计算与上一帧的移动速度(像素/帧)"""
        if self.prev_center is None:
            return 0
        dx = current_center[0] - self.prev_center[0]
        dy = current_center[1] - self.prev_center[1]
        return (dx**2 + dy**2) ** 0.5  # 欧氏距离

    def process(self, x1, y1, x2, y2):
        """处理当前预测框并返回稳定后的框坐标（增强错误过滤）"""
        # 新增：定义最大可疑框阈值（类内常量）
        self.MAX_SUSPICIOUS = 3  # 连续3次可疑则重置

        # 第一步：过滤明显无效的框（如坐标值非数字、左上角大于右下角等）
        def _is_valid_bbox(x1, y1, x2, y2):
            """校验框坐标有效性"""
            try:
                # 转换为数值类型并检查范围
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                return x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 > 0 and y2 > 0
            except (ValueError, TypeError):
                return False

        if not _is_valid_bbox(x1, y1, x2, y2):
            self.suspicious_count = getattr(self, 'suspicious_count', 0) + 1
            # 连续可疑框超过阈值，重置窗口
            if self.suspicious_count >= self.MAX_SUSPICIOUS:
                self.reset()
            # 窗口有数据时返回均值（确保返回元组），否则返回空框避免错误传播
            if self.window:
                mean_array = np.mean(self.window, axis=0)
                return (float(mean_array[0]), float(mean_array[1]),
                        float(mean_array[2]), float(mean_array[3]))
            else:
                return (0.0, 0.0, 0.0, 0.0)  # 返回空框而非原始无效值
        self.suspicious_count = 0  # 有效框重置可疑计数

        # 转换为float类型统一处理
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

        # 计算中心坐标和移动速度
        current_center = self._calc_center(x1, y1, x2, y2)
        movement_speed = self._calc_movement_speed(current_center)
        self.prev_center = current_center

        # 滑动窗口更新（只保留有效框）
        self.window.append((x1, y1, x2, y2))
        while len(self.window) > self.window_size:  # 用while确保窗口不超限（应对异常情况）
            self.window.pop(0)

        # 窗口数据不足时返回当前有效框
        if len(self.window) < 3:
            return (x1, y1, x2, y2)

        # 统计特征计算（兼容ulab的numpy）
        bbox_array = np.array(self.window, dtype=np.float)
        mean = np.mean(bbox_array, axis=0)
        std = np.std(bbox_array, axis=0)

        # 波动判断（简化逻辑并确保类型安全）
        current_box = np.array([x1, y1, x2, y2], dtype=np.float)
        deviations = abs(current_box - mean)  # 显式使用np.abs兼容ulab
        is_normal = np.all(deviations <= self.std_threshold * std)

        # 稳定策略执行
        if movement_speed > self.speed_threshold:
            return (x1, y1, x2, y2)
        elif not is_normal:
            return (float(mean[0]), float(mean[1]), float(mean[2]), float(mean[3]))
        else:
            return (x1, y1, x2, y2)

    def reset(self):
        """重置滑动窗口及状态，用于连续错误后重新跟踪"""
        self.window = []
        self.prev_center = None
        self.suspicious_count = 0


# 配置串口引脚
fpioa = FPIOA()
fpioa.set_function(11, FPIOA.UART2_TXD)
fpioa.set_function(12, FPIOA.UART2_RXD)

# 初始化UART2，波特率115200，8位数据位，无校验，1位停止位
uart = UART(UART.UART2, baudrate=115200, bits=UART.EIGHTBITS, parity=UART.PARITY_NONE, stop=UART.STOPBITS_ONE)

# 打包串口数据
def pack_data(x, y, Ring):
    x = max(0, min(x, 800))
    y = max(0, min(y, 480))
    return bytearray([
        0x2C,
        (x >> 8) & 0xFF,
        x & 0xFF,
        (y >> 8) & 0xFF,
        y & 0xFF,
        Ring,
        0x5B
    ])

def get_blob_center(blob):
    x, y, w, h = blob[0:4]
    return x + w // 2, y + h // 2

def get_middle_point(p1, p2):
    return (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2

display_mode="lcd"
if display_mode=="lcd":
    DISPLAY_WIDTH = ALIGN_UP(800, 16)
    DISPLAY_HEIGHT = 480
else:
    DISPLAY_WIDTH = ALIGN_UP(1920, 16)
    DISPLAY_HEIGHT = 1080

OUT_RGB888P_WIDTH = 800
OUT_RGB888P_HEIGH = 480
temp = -113

root_path="/sdcard/mp_deployment_source/"
config_path=root_path+"deploy_config.json"
deploy_conf={}
debug_mode=1

def two_side_pad_param(input_size,output_size):
    ratio_w = output_size[0] / input_size[0]  # 宽度缩放比例
    ratio_h = output_size[1] / input_size[1]   # 高度缩放比例
    ratio = min(ratio_w, ratio_h)  # 取较小的缩放比例
    new_w = int(ratio * input_size[0])  # 新宽度
    new_h = int(ratio * input_size[1])  # 新高度
    dw = (output_size[0] - new_w) / 2  # 宽度差
    dh = (output_size[1] - new_h) / 2  # 高度差
    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw - 0.1))
    return top, bottom, left, right,ratio

def read_deploy_config(config_path):
    # 打开JSON文件以进行读取deploy_config
    with open(config_path, 'r') as json_file:
        try:
            # 从文件中加载JSON数据
            config = ujson.load(json_file)
        except ValueError as e:
            print("JSON 解析错误:", e)
    return config

def detection():
    print("det_infer start")
    # 使用json读取内容初始化部署变量
    deploy_conf=read_deploy_config(config_path)
    kmodel_name=deploy_conf["kmodel_path"]
    labels=deploy_conf["categories"]
    confidence_threshold= deploy_conf["confidence_threshold"]
    nms_threshold = deploy_conf["nms_threshold"]
    img_size=deploy_conf["img_size"]
    num_classes=deploy_conf["num_classes"]
    color_four=get_colors(num_classes)
    nms_option = deploy_conf["nms_option"]
    model_type = deploy_conf["model_type"]
    if model_type == "AnchorBaseDet":
        anchors = deploy_conf["anchors"][0] + deploy_conf["anchors"][1] + deploy_conf["anchors"][2]
    kmodel_frame_size = img_size
    frame_size = [OUT_RGB888P_WIDTH,OUT_RGB888P_HEIGH]
    strides = [8,16,32]

    # 计算padding值
    top, bottom, left, right,ratio=two_side_pad_param(frame_size,kmodel_frame_size)

    # 初始化kpu
    kpu = nn.kpu()
    kpu.load_kmodel(root_path+kmodel_name)
    # 初始化ai2d
    ai2d = nn.ai2d()
    ai2d.set_dtype(nn.ai2d_format.NCHW_FMT,nn.ai2d_format.NCHW_FMT,np.uint8, np.uint8)
    ai2d.set_pad_param(True, [0,0,0,0,top,bottom,left,right], 0, [114,114,114])
    ai2d.set_resize_param(True, nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel )
    ai2d_builder = ai2d.build([1,3,OUT_RGB888P_HEIGH,OUT_RGB888P_WIDTH], [1,3,kmodel_frame_size[1],kmodel_frame_size[0]])
    # 初始化并配置sensor
    sensor = Sensor()
    sensor.reset()
    # 设置镜像
    sensor.set_hmirror(False)
    # 设置翻转
    sensor.set_vflip(False)
    # 通道0直接给到显示VO，格式为YUV420
    sensor.set_framesize(width = DISPLAY_WIDTH, height = DISPLAY_HEIGHT)
    sensor.set_pixformat(PIXEL_FORMAT_YUV_SEMIPLANAR_420)
    # 通道2给到AI做算法处理，格式为RGB888
    sensor.set_framesize(width = OUT_RGB888P_WIDTH , height = OUT_RGB888P_HEIGH, chn=CAM_CHN_ID_2)
    sensor.set_pixformat(PIXEL_FORMAT_RGB_888_PLANAR, chn=CAM_CHN_ID_2)
    # 绑定通道0的输出到vo
    sensor_bind_info = sensor.bind_info(x = 0, y = 0, chn = CAM_CHN_ID_0)
    Display.bind_layer(** sensor_bind_info, layer = Display.LAYER_VIDEO1)
    if display_mode=="lcd":
        # 设置为ST7701显示，默认800x480
        Display.init(Display.ST7701, to_ide = True)
    else:
        # 设置为LT9611显示，默认1920x1080
        Display.init(Display.LT9611, to_ide = True)
    #创建OSD图像
    osd_img = image.Image(DISPLAY_WIDTH, DISPLAY_HEIGHT, image.ARGB8888)
    # media初始化
    MediaManager.init()
    # 启动sensor
    sensor.run()
    rgb888p_img = None
    ai2d_input_tensor = None
    data = np.ones((1,3,kmodel_frame_size[1],kmodel_frame_size[0]),dtype=np.uint8)
    ai2d_output_tensor = nn.from_numpy(data)

    # 初始化预测框稳定器，假设最多跟踪10个目标
    max_track_objects = 10
    stabilizers = {i: BBoxStabilizer() for i in range(max_track_objects)}
    inside_start_time = None
    ring_triggered = False  # 是否已经触发
    # 在初始化区域（例如detection函数内，循环外）添加变量用于存储上一帧信息
    prev_middle_x = None  # 上一帧中心点x坐标
    prev_time = None      # 上一帧的时间戳
    speed_history = {i: [] for i in range(max_track_objects)}  # 格式: [(x, time), ...]
    target_speed = {i: 0.0 for i in range(max_track_objects)}  # 存储每100ms的速度


    while  True:
        with ScopedTiming("total",debug_mode > 0):
            rgb888p_img = sensor.snapshot(chn=CAM_CHN_ID_2)
            if rgb888p_img.format() == image.RGBP888:
                ai2d_input = rgb888p_img.to_numpy_ref()
                ai2d_input_tensor = nn.from_numpy(ai2d_input)
                # 使用ai2d进行预处理
                ai2d_builder.run(ai2d_input_tensor, ai2d_output_tensor)
                # 设置模型输入
                kpu.set_input_tensor(0, ai2d_output_tensor)
                # 模型推理
                kpu.run()
                # 获取模型输出
                results = []
                for i in range(kpu.outputs_size()):
                    out_data = kpu.get_output_tensor(i)
                    result = out_data.to_numpy()
                    result = result.reshape((result.shape[0]*result.shape[1]*result.shape[2]*result.shape[3]))
                    del out_data
                    results.append(result)
                # 使用aicube模块封装的接口进行后处理
                det_boxes = aicube.anchorbasedet_post_process( results[0], results[1], results[2], kmodel_frame_size, frame_size, strides, num_classes, confidence_threshold, nms_threshold, anchors, nms_option)
                # 绘制结果
                osd_img.clear()

                # 处理每个检测框
                if det_boxes:
                    for idx, det_boxe in enumerate(det_boxes):
                        # 只处理预设的最大目标数量以内的目标
                        if idx >= max_track_objects:
                            break

                        # 获取原始检测框坐标
                        x1, y1, x2, y2 = det_boxe[2], det_boxe[3], det_boxe[4], det_boxe[5]

                        # 使用稳定器处理检测框
                        stable_x1, stable_y1, stable_x2, stable_y2 = stabilizers[idx].process(x1, y1, x2, y2)

                        # 转换坐标到显示尺寸
                        x = int(stable_x1 * DISPLAY_WIDTH // OUT_RGB888P_WIDTH)
                        y = int(stable_y1 * DISPLAY_HEIGHT // OUT_RGB888P_HEIGH)
                        w = int((stable_x2 - stable_x1) * DISPLAY_WIDTH // OUT_RGB888P_WIDTH)
                        h = int((stable_y2 - stable_y1) * DISPLAY_HEIGHT // OUT_RGB888P_HEIGH)

                        # 计算稳定后的中心坐标
                        middle_x = int((stable_x2 - stable_x1) // 2 + stable_x1)
                        middle_y = int((stable_y2 - stable_y1) // 2 + stable_y1 + temp)

                        # Ring触发逻辑（保持不变）
                        if not ring_triggered:
                            if inside_start_time is None:
                                inside_start_time = time.ticks_ms()
                            if time.ticks_diff(time.ticks_ms(), inside_start_time)>3000:
                                elapsed=4000
                                if elapsed>3000:
                                    if stable_x1<400 and stable_x2>400 and stable_y1<240 and stable_y2>240:
                                        Ring = 1
                                    else:
                                        Ring = 0
                            else:
                                Ring = 0
                        else:
                            Ring = 1

                        # 计算横向x轴移动速度（像素/毫秒）
                        current_time = time.ticks_ms()

                        # 存储当前位置和时间到历史记录
                        speed_history[idx].append((middle_x, current_time))

                        # 过滤100ms以外的旧数据（只保留最近100ms内的记录）
                        # 计算100ms前的时间戳
                        cutoff_time = current_time - 100
                        # 保留时间戳 >= cutoff_time 的记录
                        speed_history[idx] = [(x, t) for x, t in speed_history[idx] if t >= cutoff_time]

                        # 当有足够的历史数据（至少2个点，且时间跨度 >= 100ms）时计算速度
                        if len(speed_history[idx]) >= 2:
                            # 取最早和最新的记录计算总位移和总时间
                            first_x, first_t = speed_history[idx][0]
                            last_x, last_t = speed_history[idx][-1]
                            time_diff = last_t - first_t  # 时间差（毫秒）

                            if time_diff >= 100:  # 确保时间跨度至少100ms
                                x_diff = last_x - first_x  # x方向总位移
                                # 计算100ms内的平均速度（像素/毫秒）
                                target_speed[idx] = x_diff / time_diff
                                # 清空历史记录，开始下一个100ms的计算
                                speed_history[idx] = [(last_x, last_t)]  # 保留最后一个点作为下一轮的起点
                            else:
                                # 时间不足100ms时沿用之前的速度
                                pass
                        else:
                            # 数据不足时速度为0
                            target_speed[idx] = 0.0

                        # 打印每100ms的速度信息
                        print(f"目标{idx}每100ms横向x轴速度: {target_speed[idx]*150:.6f}像素/毫秒")
                        if w*h<30000:
                            middle_x+=int(target_speed[idx]*150)

                        # 串口发送
                        try:
                            data = pack_data(middle_x, middle_y, Ring)
                            uart.write(data)
                            print(f"已发送坐标: ({middle_x},{middle_y},{Ring}面积{w*h})")
                        except Exception as e:
                            print(f"串口发送错误: {e}")

                        # 绘制稳定后的框
                        osd_img.draw_rectangle(x, y, w, h, color=color_four[det_boxe[0]][1:])
                        text = labels[det_boxe[0]] + " " + str(round(det_boxe[1], 2))
                        osd_img.draw_string_advanced(x, y-40, 32, text, color=color_four[det_boxe[0]][1:])

                # 重置未使用的稳定器缓冲区
                for i in range(len(det_boxes), max_track_objects):
                    stabilizers[i].reset()

                Display.show_image(osd_img, 0, 0, Display.LAYER_OSD3)
                gc.collect()
            rgb888p_img = None
    del ai2d_input_tensor
    del ai2d_output_tensor
    #停止摄像头输出
    sensor.stop()
    #去初始化显示设备
    Display.deinit()
    #释放媒体缓冲区
    MediaManager.deinit()
    gc.collect()
    time.sleep(1)
    nn.shrink_memory_pool()
    print("det_infer end")
    return 0

if __name__=="__main__":
    detection()
