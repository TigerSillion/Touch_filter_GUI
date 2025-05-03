#!/usr/bin/env python3
"""
touch_filter_gui.py
触摸数据可视化和滤波工具的GUI版本
"""

import sys
import os
import traceback
import logging
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
matplotlib.rcParams['font.size'] = 12  # 设置默认字体大小

# 配置日志
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler()])

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    import numpy as np
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QComboBox, QLabel, 
                               QFileDialog, QSpinBox, QDoubleSpinBox, QGroupBox,
                               QMessageBox, QListWidget, QCheckBox, QTextEdit,
                               QTabWidget, QScrollArea, QLineEdit, QCheckBox, QSplitter,
                               QColorDialog, QListWidgetItem, QMenu)
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont, QColor
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from scipy.signal import butter, filtfilt, medfilt, welch, savgol_filter
    from scipy.stats import kurtosis, skew
    logger.info("所有依赖包导入成功")
except ImportError as e:
    logger.error(f"导入错误: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

class WaveformAnalyzer:
    """波形分析器类"""
    def __init__(self):
        self.analysis_results = {}
        
    def analyze_waveform(self, data):
        """分析波形特性"""
        try:
            # 基本统计特性
            mean = np.mean(data)
            std = np.std(data)
            max_val = np.max(data)
            min_val = np.min(data)
            range_val = max_val - min_val
            kurt = kurtosis(data)
            skewness = skew(data)
            
            # 频率特性
            fs = 1000  # 假设采样率为1000Hz
            f, Pxx = welch(data, fs=fs, nperseg=1024)
            dominant_freq = f[np.argmax(Pxx)]
            
            # 噪声水平估计
            noise_level = std / mean if mean != 0 else std
            
            result = {
                '基本统计': {
                    '均值': mean,
                    '标准差': std,
                    '最大值': max_val,
                    '最小值': min_val,
                    '范围': range_val,
                    '峰度': kurt,
                    '偏度': skewness
                },
                '频率特性': {
                    '主频': dominant_freq,
                    '采样率': fs
                },
                '噪声水平': noise_level
            }
            
            return result
        except Exception as e:
            logger.error(f"波形分析错误: {e}")
            return None
            
    def recommend_filter(self, analysis_result):
        """根据分析结果推荐滤波器"""
        if not analysis_result:
            return "raw"
            
        noise_level = analysis_result['噪声水平']
        dominant_freq = analysis_result['频率特性']['主频']
        
        if noise_level > 0.1:  # 高噪声
            if dominant_freq < 10:  # 低频信号
                return "butter"
            else:  # 高频信号
                return "median"
        else:  # 低噪声
            if dominant_freq < 10:  # 低频信号
                return "ema"
            else:  # 高频信号
                return "moving_avg"

class CustomFilter:
    """自定义滤波器类"""
    def __init__(self):
        self.formula = ""
        self.parameters = {}
        
    def set_formula(self, formula):
        """设置滤波器公式"""
        self.formula = formula
        
    def set_parameters(self, **kwargs):
        """设置滤波器参数"""
        self.parameters.update(kwargs)
        
    def apply_filter(self, data):
        """应用自定义滤波器"""
        try:
            # 创建局部变量字典
            local_vars = {'data': data, 'np': np}
            local_vars.update(self.parameters)
            
            # 执行公式
            result = eval(self.formula, {'__builtins__': None}, local_vars)
            return result
        except Exception as e:
            logger.error(f"自定义滤波器错误: {e}")
            return data

class WaveformStyle:
    """波形样式类"""
    def __init__(self, color='blue', line_style='-', line_width=1.5, highlighted=False):
        self.color = color
        self.line_style = line_style
        self.line_width = line_width
        self.highlighted = highlighted

class WaveformData:
    """波形数据管理类"""
    def __init__(self):
        self.raw_data = None
        self.filtered_data = {}
        self.selected_columns = set()  # 选中的波形列名
        self.colors = {}  # 每个波形的颜色
        self.analyzer = WaveformAnalyzer()  # 波形分析器
        self.custom_filter = CustomFilter()  # 自定义滤波器
        self.normalized_data = {}  # 归一化后的数据
        self.active_filters = []  # 当前激活的滤波器列表
        self.status_data = None  # 状态数据
        self.styles = {}  # 波形样式字典

    def normalize_status(self, data):
        """归一化状态数据"""
        return (data - data.min()) / (data.max() - data.min() + 1e-10)

    def apply_cascade_filters(self, data):
        """应用级联滤波器"""
        result = data.copy()
        for filter_type, params in self.active_filters:
            if filter_type == 'moving_average':
                result = TouchFilter.moving_average(result, params['window_size'])
            elif filter_type == 'median_filter':
                result = TouchFilter.median_filter(result, params['window_size'])
            elif filter_type == 'exponential_moving_average':
                result = TouchFilter.exponential_moving_average(result, params['alpha'])
            elif filter_type == 'butterworth_filter':
                result = TouchFilter.butterworth_filter(result, params['cutoff'], 
                                                      params['fs'], params['order'])
            elif filter_type == 'savitzky_golay':
                result = TouchFilter.savitzky_golay(result, params['window_size'], 
                                                  params['order'])
            elif filter_type == 'threshold_filter':
                result = TouchFilter.threshold_filter(result, params['threshold'])
            elif filter_type == 'rate_limit_filter':
                result = TouchFilter.rate_limit_filter(result, params['max_rate'])
            elif filter_type == 'group_average_filter':
                result = TouchFilter.group_average_filter(result, params['group_size'])
        return result
    
    def get_filter_code(self, filter_type, params, language='python'):
        """获取滤波器代码，支持多种语言"""
        if language == 'python':
            return self._get_python_filter_code(filter_type, params)
        elif language == 'c':
            return self._get_c_filter_code(filter_type, params)
        elif language == 'excel':
            return self._get_excel_filter_code(filter_type, params)
        else:
            return "不支持的语言"
    
    def _get_python_filter_code(self, filter_type, params):
        """获取Python格式的滤波器代码"""
        if filter_type == 'moving_average':
            return f"def moving_average(data, window_size={params['window_size']}):\n    return np.convolve(data, np.ones({params['window_size']})/float({params['window_size']}), 'same')"
        elif filter_type == 'median_filter':
            return f"def median_filter(data, window_size={params['window_size']}):\n    return medfilt(data, kernel_size={params['window_size']})"
        elif filter_type == 'exponential_moving_average':
            return f"def exponential_moving_average(data, alpha={params['alpha']}):\n    ema = np.zeros_like(data)\n    ema[0] = data[0]\n    for i in range(1, len(data)):\n        ema[i] = {params['alpha']} * data[i] + (1 - {params['alpha']}) * ema[i-1]\n    return ema"
        elif filter_type == 'butterworth_filter':
            return f"def butterworth_filter(data, cutoff={params['cutoff']}, fs={params['fs']}, order={params['order']}):\n    nyq = 0.5 * {params['fs']}\n    normal_cutoff = {params['cutoff']} / nyq\n    b, a = butter({params['order']}, normal_cutoff, btype='low', analog=False)\n    return filtfilt(b, a, data)"
        elif filter_type == 'savitzky_golay':
            return f"def savitzky_golay(data, window_size={params['window_size']}, order={params['order']}):\n    return savgol_filter(data, {params['window_size']}, {params['order']})"
        elif filter_type == 'threshold_filter':
            return f"def threshold_filter(data, threshold={params['threshold']}):\n    filtered = np.copy(data)\n    for i in range(1, len(data)):\n        if abs(data[i] - data[i-1]) > {params['threshold']}:\n            filtered[i] = data[i-1]\n    return filtered"
        elif filter_type == 'rate_limit_filter':
            return f"def rate_limit_filter(data, max_rate={params['max_rate']}):\n    filtered = np.copy(data)\n    for i in range(1, len(data)):\n        rate = abs(data[i] - data[i-1])\n        if rate > {params['max_rate']}:\n            filtered[i] = filtered[i-1] + np.sign(data[i] - data[i-1]) * {params['max_rate']}\n    return filtered"
        elif filter_type == 'group_average_filter':
            return f"def group_average_filter(data, group_size={params['group_size']}):\n    filtered = np.copy(data)\n    for i in range(0, len(data), {params['group_size']}):\n        group = data[i:i+{params['group_size']}]\n        if len(group) >= 4:\n            group = np.sort(group)[1:-1]\n            filtered[i:i+len(group)] = np.mean(group)\n    return filtered"
        else:
            return "# 未知滤波器类型"
    
    def _get_c_filter_code(self, filter_type, params):
        """获取C语言格式的滤波器代码"""
        if filter_type == 'moving_average':
            return f"""void moving_average(float *data, float *result, int data_len, int window_size) {{
    // 移动平均滤波
    window_size = {params['window_size']}; 
    for (int i = 0; i < data_len; i++) {{
        float sum = 0.0f;
        int count = 0;
        for (int j = i - window_size / 2; j <= i + window_size / 2; j++) {{
            if (j >= 0 && j < data_len) {{
                sum += data[j];
                count++;
            }}
        }}
        result[i] = sum / count;
    }}
}}"""
        elif filter_type == 'median_filter':
            return f"""void median_filter(float *data, float *result, int data_len, int window_size) {{
    // 中值滤波
    window_size = {params['window_size']};
    for (int i = 0; i < data_len; i++) {{
        float window[{params['window_size']}];
        int count = 0;
        for (int j = i - window_size / 2; j <= i + window_size / 2; j++) {{
            if (j >= 0 && j < data_len) {{
                window[count++] = data[j];
            }}
        }}
        // 对窗口内数据排序
        for (int j = 0; j < count; j++) {{
            for (int k = j + 1; k < count; k++) {{
                if (window[j] > window[k]) {{
                    float temp = window[j];
                    window[j] = window[k];
                    window[k] = temp;
                }}
            }}
        }}
        // 取中值
        result[i] = window[count / 2];
    }}
}}"""
        elif filter_type == 'exponential_moving_average':
            return f"""void exponential_moving_average(float *data, float *result, int data_len, float alpha) {{
    // 指数移动平均
    alpha = {params['alpha']};
    result[0] = data[0];
    for (int i = 1; i < data_len; i++) {{
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1];
    }}
}}"""
        else:
            return "// 此滤波器的C语言代码较复杂，请参考Python版本";
    
    def _get_excel_filter_code(self, filter_type, params):
        """获取Excel公式格式的滤波器代码"""
        if filter_type == 'moving_average':
            return f"=AVERAGE(OFFSET(A1,ROW()-1-FLOOR({params['window_size']}/2,1),0,{params['window_size']},1))"
        elif filter_type == 'median_filter':
            return f"=MEDIAN(OFFSET(A1,ROW()-1-FLOOR({params['window_size']}/2,1),0,{params['window_size']},1))"
        elif filter_type == 'exponential_moving_average':
            return f"=IF(ROW()=1,A1,{params['alpha']}*A1+(1-{params['alpha']})*B0)"
        else:
            return "此滤波器无法使用Excel公式简单表示";

class TouchFilter:
    """触摸数据滤波器类"""
    @staticmethod
    def moving_average(data, window_size=5):
        """移动平均滤波"""
        return np.convolve(data, np.ones(window_size)/window_size, 'same')
    
    @staticmethod
    def median_filter(data, window_size=5):
        """中值滤波"""
        return medfilt(data, kernel_size=window_size)
    
    @staticmethod
    def exponential_moving_average(data, alpha=0.1):
        """指数移动平均"""
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema
    
    @staticmethod
    def butterworth_filter(data, cutoff=5.0, fs=60.0, order=3):
        """巴特沃斯滤波"""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)
    
    @staticmethod
    def savitzky_golay(data, window_size=5, order=2):
        """Savitzky-Golay滤波"""
        return savgol_filter(data, window_size, order)
    
    @staticmethod
    def threshold_filter(data, threshold=100):
        """限幅滤波"""
        filtered = np.copy(data)
        for i in range(1, len(data)):
            if abs(data[i] - data[i-1]) > threshold:
                filtered[i] = data[i-1]
        return filtered
    
    @staticmethod
    def rate_limit_filter(data, max_rate=10):
        """增幅限制滤波"""
        filtered = np.copy(data)
        for i in range(1, len(data)):
            rate = abs(data[i] - data[i-1])
            if rate > max_rate:
                filtered[i] = filtered[i-1] + np.sign(data[i] - data[i-1]) * max_rate
        return filtered
    
    @staticmethod
    def group_average_filter(data, group_size=6):
        """分组平均滤波"""
        filtered = np.copy(data)
        for i in range(0, len(data), group_size):
            group = data[i:i+group_size]
            if len(group) >= 4:  # 至少需要4个数据
                # 移除最大值和最小值
                group = np.sort(group)[1:-1]
                # 计算平均值
                filtered[i:i+len(group)] = np.mean(group)
        return filtered

class TouchFilterGUI(QMainWindow):
    def __init__(self):
        try:
            logger.info("初始化GUI...")
            super().__init__()
            self.setWindowTitle('触摸数据滤波分析工具')
            self.setGeometry(100, 100, 1600, 900)  # 增加窗口大小
            
            # 设置全局字体
            font = QFont("Microsoft YaHei", 10)
            QApplication.setFont(font)
            
            # 数据存储
            self.waveform_data = WaveformData()
            
            # 创建主窗口部件
            splitter = QSplitter()
            self.setCentralWidget(splitter)
            
            # 左侧控制面板
            left_panel = QWidget()
            left_layout = QVBoxLayout(left_panel)
            
            # 文件操作组
            file_group = QGroupBox("文件操作")
            file_layout = QVBoxLayout()
            self.load_button = QPushButton("加载CSV文件")
            self.load_button.setFont(font)
            self.load_button.clicked.connect(self.load_csv)
            file_layout.addWidget(self.load_button)
            file_group.setLayout(file_layout)
            left_layout.addWidget(file_group)
            
            # 波形选择组
            waveform_group = QGroupBox("波形选择")
            waveform_layout = QVBoxLayout()
            self.waveform_list = QListWidget()
            self.waveform_list.setFont(font)
            self.waveform_list.setSelectionMode(QListWidget.MultiSelection)
            self.waveform_list.itemSelectionChanged.connect(self.update_plot)
            waveform_layout.addWidget(self.waveform_list)
            waveform_group.setLayout(waveform_layout)
            left_layout.addWidget(waveform_group)
            
            # 状态选择组
            status_group = QGroupBox("状态选择")
            status_layout = QVBoxLayout()
            self.status_list = QListWidget()
            self.status_list.setFont(font)
            self.status_list.setSelectionMode(QListWidget.SingleSelection)
            self.status_list.itemSelectionChanged.connect(self.update_plot)
            status_layout.addWidget(self.status_list)
            status_group.setLayout(status_layout)
            left_layout.addWidget(status_group)
            
            # 坐标范围设置组
            range_group = QGroupBox("坐标范围设置")
            range_layout = QVBoxLayout()
            
            # X轴范围
            x_range_layout = QHBoxLayout()
            x_range_layout.addWidget(QLabel("X轴范围:"))
            self.x_min = QLineEdit("")
            self.x_min.setPlaceholderText("最小值")
            self.x_max = QLineEdit("")
            self.x_max.setPlaceholderText("最大值")
            x_range_layout.addWidget(self.x_min)
            x_range_layout.addWidget(QLabel("到"))
            x_range_layout.addWidget(self.x_max)
            range_layout.addLayout(x_range_layout)
            
            # Y轴范围
            y_range_layout = QHBoxLayout()
            y_range_layout.addWidget(QLabel("Y轴范围:"))
            self.y_min = QLineEdit("")
            self.y_min.setPlaceholderText("最小值")
            self.y_max = QLineEdit("")
            self.y_max.setPlaceholderText("最大值")
            y_range_layout.addWidget(self.y_min)
            y_range_layout.addWidget(QLabel("到"))
            y_range_layout.addWidget(self.y_max)
            range_layout.addLayout(y_range_layout)
            
            # 应用坐标范围按钮
            apply_range_layout = QHBoxLayout()
            self.apply_range_btn = QPushButton("应用坐标范围")
            self.apply_range_btn.clicked.connect(self.apply_axis_range)
            apply_range_layout.addWidget(self.apply_range_btn)
            
            # 重置坐标范围按钮
            self.reset_range_btn = QPushButton("重置坐标范围")
            self.reset_range_btn.clicked.connect(self.reset_zoom)
            apply_range_layout.addWidget(self.reset_range_btn)
            range_layout.addLayout(apply_range_layout)
            
            range_group.setLayout(range_layout)
            left_layout.addWidget(range_group)
            
            # 滤波器设置组
            filter_group = QGroupBox("滤波设置")
            filter_layout = QVBoxLayout()
            
            # 滤波器类型选择
            filter_type_layout = QHBoxLayout()
            self.filter_combo = QComboBox()
            self.filter_combo.setFont(font)
            self.filter_combo.addItems([
                'raw', 'moving_average', 'median_filter', 'exponential_moving_average',
                'butterworth_filter', 'savitzky_golay', 'threshold_filter',
                'rate_limit_filter', 'group_average_filter', 'custom'
            ])
            self.filter_combo.currentTextChanged.connect(self.update_filter_params)
            filter_type_layout.addWidget(QLabel("滤波类型:"))
            filter_type_layout.addWidget(self.filter_combo)
            filter_layout.addLayout(filter_type_layout)
            
            # 添加滤波器按钮
            self.add_filter_btn = QPushButton("添加滤波器")
            self.add_filter_btn.setFont(font)
            self.add_filter_btn.clicked.connect(self.add_filter)
            filter_layout.addWidget(self.add_filter_btn)
            
            # 滤波器列表
            self.filter_list = QListWidget()
            self.filter_list.setFont(font)
            self.filter_list.setSelectionMode(QListWidget.SingleSelection)
            self.filter_list.itemSelectionChanged.connect(self.update_plot)
            filter_layout.addWidget(self.filter_list)
            
            # 移除滤波器按钮
            self.remove_filter_btn = QPushButton("移除选中滤波器")
            self.remove_filter_btn.setFont(font)
            self.remove_filter_btn.clicked.connect(self.remove_filter)
            filter_layout.addWidget(self.remove_filter_btn)
            
            # 初始化滤波器参数控件
            self.param_widgets = {}
            
            # 移动平均参数
            ma_group = QGroupBox("移动平均参数")
            ma_layout = QVBoxLayout()
            self.ma_window = QLineEdit("5")
            ma_layout.addWidget(QLabel("窗口大小:"))
            ma_layout.addWidget(self.ma_window)
            ma_group.setLayout(ma_layout)
            filter_layout.addWidget(ma_group)
            self.param_widgets['moving_average'] = ma_group
            
            # 中值滤波参数
            median_group = QGroupBox("中值滤波参数")
            median_layout = QVBoxLayout()
            self.median_window = QLineEdit("5")
            median_layout.addWidget(QLabel("窗口大小:"))
            median_layout.addWidget(self.median_window)
            median_group.setLayout(median_layout)
            filter_layout.addWidget(median_group)
            self.param_widgets['median_filter'] = median_group
            
            # 指数移动平均参数
            ema_group = QGroupBox("指数移动平均参数")
            ema_layout = QVBoxLayout()
            self.ema_alpha = QLineEdit("0.1")
            ema_layout.addWidget(QLabel("Alpha值:"))
            ema_layout.addWidget(self.ema_alpha)
            ema_group.setLayout(ema_layout)
            filter_layout.addWidget(ema_group)
            self.param_widgets['exponential_moving_average'] = ema_group
            
            # 巴特沃斯滤波参数
            butter_group = QGroupBox("巴特沃斯滤波参数")
            butter_layout = QVBoxLayout()
            self.butter_cutoff = QLineEdit("5.0")
            self.butter_fs = QLineEdit("60.0")
            self.butter_order = QLineEdit("3")
            butter_layout.addWidget(QLabel("截止频率:"))
            butter_layout.addWidget(self.butter_cutoff)
            butter_layout.addWidget(QLabel("采样频率:"))
            butter_layout.addWidget(self.butter_fs)
            butter_layout.addWidget(QLabel("阶数:"))
            butter_layout.addWidget(self.butter_order)
            butter_group.setLayout(butter_layout)
            filter_layout.addWidget(butter_group)
            self.param_widgets['butterworth_filter'] = butter_group
            
            # Savitzky-Golay滤波参数
            sg_group = QGroupBox("Savitzky-Golay滤波参数")
            sg_layout = QVBoxLayout()
            self.sg_window = QLineEdit("5")
            self.sg_order = QLineEdit("2")
            sg_layout.addWidget(QLabel("窗口大小:"))
            sg_layout.addWidget(self.sg_window)
            sg_layout.addWidget(QLabel("阶数:"))
            sg_layout.addWidget(self.sg_order)
            sg_group.setLayout(sg_layout)
            filter_layout.addWidget(sg_group)
            self.param_widgets['savitzky_golay'] = sg_group
            
            # 限幅滤波参数
            threshold_group = QGroupBox("限幅滤波参数")
            threshold_layout = QVBoxLayout()
            self.threshold_value = QLineEdit("100")
            threshold_layout.addWidget(QLabel("阈值:"))
            threshold_layout.addWidget(self.threshold_value)
            threshold_group.setLayout(threshold_layout)
            filter_layout.addWidget(threshold_group)
            self.param_widgets['threshold_filter'] = threshold_group
            
            # 增幅限制滤波参数
            rate_limit_group = QGroupBox("增幅限制滤波参数")
            rate_limit_layout = QVBoxLayout()
            self.max_rate = QLineEdit("10")
            rate_limit_layout.addWidget(QLabel("最大变化率:"))
            rate_limit_layout.addWidget(self.max_rate)
            rate_limit_group.setLayout(rate_limit_layout)
            filter_layout.addWidget(rate_limit_group)
            self.param_widgets['rate_limit_filter'] = rate_limit_group
            
            # 分组平均滤波参数
            group_avg_group = QGroupBox("分组平均滤波参数")
            group_avg_layout = QVBoxLayout()
            self.group_size = QLineEdit("6")
            group_avg_layout.addWidget(QLabel("组大小:"))
            group_avg_layout.addWidget(self.group_size)
            group_avg_group.setLayout(group_avg_layout)
            filter_layout.addWidget(group_avg_group)
            self.param_widgets['group_average_filter'] = group_avg_group
            
            # 自定义滤波器设置
            custom_filter_group = QGroupBox("自定义滤波器")
            custom_filter_layout = QVBoxLayout()
            self.formula_edit = QTextEdit()
            self.formula_edit.setPlaceholderText("输入Python表达式，例如：\n"
                                               "np.convolve(data, np.ones(5)/5, 'same')")
            custom_filter_layout.addWidget(self.formula_edit)
            custom_filter_group.setLayout(custom_filter_layout)
            filter_layout.addWidget(custom_filter_group)
            self.param_widgets['custom'] = custom_filter_group
            
            # 初始化参数控件，只显示当前选择的滤波器参数
            self.update_filter_params()
            
            filter_group.setLayout(filter_layout)
            left_layout.addWidget(filter_group)
            
            # 波形样式设置组
            style_group = QGroupBox("波形样式设置")
            style_layout = QVBoxLayout()
            
            # 波形列表
            self.style_list = QListWidget()
            self.style_list.setFont(font)
            self.style_list.setContextMenuPolicy(Qt.CustomContextMenu)
            self.style_list.customContextMenuRequested.connect(self.show_style_menu)
            style_layout.addWidget(self.style_list)
            
            # 高亮选中波形按钮
            self.highlight_btn = QPushButton("高亮选中波形")
            self.highlight_btn.setFont(font)
            self.highlight_btn.clicked.connect(self.highlight_selected_waveform)
            style_layout.addWidget(self.highlight_btn)
            
            style_group.setLayout(style_layout)
            left_layout.addWidget(style_group)
            
            # 滤波器信息组
            filter_info_group = QGroupBox("滤波器信息")
            filter_info_layout = QVBoxLayout()
            
            # 滤波器代码选择
            code_layout = QHBoxLayout()
            self.code_language = QComboBox()
            self.code_language.addItems(["python", "c", "excel"])
            self.code_language.currentTextChanged.connect(self.update_filter_code)
            code_layout.addWidget(QLabel("代码语言:"))
            code_layout.addWidget(self.code_language)
            filter_info_layout.addLayout(code_layout)
            
            # 滤波器代码显示
            self.filter_code_text = QTextEdit()
            self.filter_code_text.setFont(QFont("Courier New", 9))
            self.filter_code_text.setReadOnly(True)
            filter_info_layout.addWidget(self.filter_code_text)
            
            filter_info_group.setLayout(filter_info_layout)
            left_layout.addWidget(filter_info_group)
            
            splitter.addWidget(left_panel)
            
            # 右侧图表区域
            right_panel = QWidget()
            right_layout = QVBoxLayout(right_panel)
            
            # 创建图表
            self.figure = Figure(figsize=(12, 6))
            self.canvas = FigureCanvas(self.figure)
            self.canvas.mpl_connect('scroll_event', self.on_scroll)  # 添加滚轮缩放支持
            self.canvas.mpl_connect('button_press_event', self.on_press)
            self.canvas.mpl_connect('button_release_event', self.on_release)
            self.canvas.mpl_connect('motion_notify_event', self.on_motion)
            right_layout.addWidget(self.canvas)
            
            # 重置缩放按钮
            self.reset_zoom_btn = QPushButton("重置缩放")
            self.reset_zoom_btn.setFont(font)
            self.reset_zoom_btn.clicked.connect(self.reset_zoom)
            right_layout.addWidget(self.reset_zoom_btn)
            
            # 初始化图表
            self.ax = self.figure.add_subplot(111)
            self.ax2 = self.ax.twinx()  # 创建第二个Y轴用于状态显示
            
            # 拖动相关变量
            self._dragging = False
            self._last_xy = None
            
            splitter.addWidget(right_panel)
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)
            
            logger.info("GUI初始化完成")
        except Exception as e:
            logger.error(f"GUI初始化错误: {e}")
            logger.error(traceback.format_exc())
            raise

    def on_scroll(self, event):
        """优化缩放体验：滚轮缩放X轴，Shift+滚轮缩放Y轴"""
        try:
            logger.info(f"滚轮事件: {event.button} {'带Shift键' if event.key == 'shift' else '不带修饰键'}")
            
            # 检查事件是否在图表中
            if event.inaxes is None:
                logger.info("滚轮事件不在图表中，忽略")
                return
            
            logger.info(f"事件坐标: x={event.xdata}, y={event.ydata}")
            
            base_scale = 1.1
            ax = self.ax
            
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            
            xdata = event.xdata
            ydata = event.ydata
            
            logger.info(f"当前X轴范围: {cur_xlim}, Y轴范围: {cur_ylim}")
            
            if event.button == 'up':
                scale_factor = 1/base_scale
                logger.info("缩小视图")
            elif event.button == 'down':
                scale_factor = base_scale
                logger.info("放大视图")
            else:
                scale_factor = 1
                logger.info("未知滚轮事件，保持比例")
                
            if event.key == 'shift':
                # 只缩放Y轴
                logger.info("执行Y轴缩放")
                new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
                rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])
                new_ylim = [ydata - new_height * (1-rely), ydata + new_height * rely]
                logger.info(f"新的Y轴范围: {new_ylim}")
                ax.set_ylim(new_ylim)
            else:
                # 只缩放X轴
                logger.info("执行X轴缩放")
                new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
                relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
                new_xlim = [xdata - new_width * (1-relx), xdata + new_width * relx]
                logger.info(f"新的X轴范围: {new_xlim}")
                ax.set_xlim(new_xlim)
                
            logger.info("刷新画布")
            self.canvas.draw_idle()  # 使用draw_idle而不是draw，提高性能
            logger.info("缩放完成")
            
        except Exception as e:
            logger.error(f"滚轮缩放错误: {e}")
            logger.error(traceback.format_exc())

    def on_press(self, event):
        try:
            if event.button == 2:  # 鼠标中键
                logger.info("鼠标中键按下，开始拖动")
                self._dragging = True
                self._last_xy = (event.x, event.y)
                logger.info(f"记录初始位置: x={event.x}, y={event.y}")
                
                # 改变鼠标光标为抓手
                self.canvas.setCursor(Qt.ClosedHandCursor)
        except Exception as e:
            logger.error(f"鼠标按下事件错误: {e}")
            logger.error(traceback.format_exc())

    def on_release(self, event):
        try:
            if self._dragging:
                logger.info("释放鼠标，结束拖动")
                self._dragging = False
                self._last_xy = None
                
                # 恢复鼠标光标
                self.canvas.setCursor(Qt.ArrowCursor)
        except Exception as e:
            logger.error(f"鼠标释放事件错误: {e}")
            logger.error(traceback.format_exc())

    def on_motion(self, event):
        try:
            if not self._dragging:
                return
                
            logger.info(f"拖动事件: x={event.x}, y={event.y}")
            
            # 如果事件不在坐标轴内，忽略
            if event.inaxes != self.ax:
                logger.info("拖动超出图表范围，忽略")
                return
                
            dx = event.x - self._last_xy[0]
            dy = event.y - self._last_xy[1]
            
            logger.info(f"鼠标移动: dx={dx}, dy={dy}")
            
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()
            
            # 计算平移比例
            x_range = cur_xlim[1] - cur_xlim[0]
            y_range = cur_ylim[1] - cur_ylim[0]
            
            # 这里的0.005可根据实际体验调整，原来的0.01可能太大导致移动太快
            x_scale = 0.005
            y_scale = 0.005
            
            new_xlim = (cur_xlim[0] - dx * x_range * x_scale, 
                        cur_xlim[1] - dx * x_range * x_scale)
            new_ylim = (cur_ylim[0] + dy * y_range * y_scale, 
                        cur_ylim[1] + dy * y_range * y_scale)
            
            logger.info(f"新的X轴范围: {new_xlim}, Y轴范围: {new_ylim}")
            
            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            self._last_xy = (event.x, event.y)
            
            self.canvas.draw_idle()  # 使用draw_idle而不是draw，提高响应速度
            logger.info("拖动更新完成")
        except Exception as e:
            logger.error(f"拖动事件错误: {e}")
            logger.error(traceback.format_exc())

    def reset_zoom(self):
        try:
            logger.info("重置缩放")
            self.ax.relim()
            self.ax.autoscale_view()
            
            # 确保状态显示区域的Y轴范围正确
            self.ax2.set_ylim(-0.1, 1.1)
            
            self.canvas.draw()
            logger.info("重置缩放完成")
        except Exception as e:
            logger.error(f"重置缩放错误: {e}")
            logger.error(traceback.format_exc())

    def add_filter(self):
        """添加滤波器到级联列表"""
        try:
            logger.info("开始添加滤波器")
            filter_type = self.filter_combo.currentText()
            logger.info(f"当前选择的滤波器类型: {filter_type}")
            
            if filter_type == 'raw':
                logger.info("原始数据滤波器，不添加")
                return
                
            # 获取当前滤波器参数
            logger.info("开始获取滤波器参数")
            params = self.get_current_filter_params()
            logger.info(f"获取到的参数: {params}")
            
            if params is None:
                logger.warning("参数获取失败，不添加滤波器")
                return
                
            # 添加到滤波器列表
            logger.info("添加滤波器到列表")
            self.waveform_data.active_filters.append((filter_type, params))
            self.filter_list.addItem(f"{filter_type}: {params}")
            
            # 更新滤波器代码显示
            self.update_filter_code()
            
            self.update_plot()
            logger.info("添加滤波器流程完成")
        except Exception as e:
            logger.error(f"添加滤波器错误: {e}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "错误", f"添加滤波器时发生错误：\n{str(e)}")

    def remove_filter(self):
        """从级联列表中移除滤波器"""
        current_row = self.filter_list.currentRow()
        if current_row >= 0:
            self.waveform_data.active_filters.pop(current_row)
            self.filter_list.takeItem(current_row)
            self.update_plot()

    def get_current_filter_params(self):
        """获取当前滤波器的参数，增加健壮性校验，防止闪退"""
        filter_type = self.filter_combo.currentText()
        logger.info(f"获取'{filter_type}'滤波器参数")
        
        try:
            if filter_type == 'moving_average':
                logger.info(f"处理移动平均滤波参数，控件是否存在: {hasattr(self, 'ma_window')}")
                if not hasattr(self, 'ma_window'):
                    raise ValueError('移动平均窗口控件未初始化')
                    
                val = self.ma_window.text()
                logger.info(f"获取到窗口大小参数: {val}")
                
                if not val.isdigit() or int(val) < 1:
                    raise ValueError('窗口大小需为正整数')
                return {'window_size': int(val)}
                
            elif filter_type == 'median_filter':
                logger.info(f"处理中值滤波参数，控件是否存在: {hasattr(self, 'median_window')}")
                if not hasattr(self, 'median_window'):
                    raise ValueError('中值滤波窗口控件未初始化')
                    
                val = self.median_window.text()
                logger.info(f"获取到窗口大小参数: {val}")
                
                if not val.isdigit() or int(val) < 1:
                    raise ValueError('窗口大小需为正整数')
                return {'window_size': int(val)}
                
            elif filter_type == 'exponential_moving_average':
                logger.info(f"处理指数移动平均参数，控件是否存在: {hasattr(self, 'ema_alpha')}")
                if not hasattr(self, 'ema_alpha'):
                    raise ValueError('指数移动平均Alpha控件未初始化')
                    
                val = self.ema_alpha.text()
                logger.info(f"获取到Alpha参数: {val}")
                
                alpha = float(val)
                if not (0 < alpha < 1):
                    raise ValueError('Alpha值需在0和1之间')
                return {'alpha': alpha}
                
            elif filter_type == 'butterworth_filter':
                logger.info(f"处理巴特沃斯滤波参数")
                if not all(hasattr(self, attr) for attr in ['butter_cutoff', 'butter_fs', 'butter_order']):
                    raise ValueError('巴特沃斯滤波参数控件未初始化')
                    
                cutoff = float(self.butter_cutoff.text())
                fs = float(self.butter_fs.text())
                order = int(self.butter_order.text())
                logger.info(f"获取到参数: cutoff={cutoff}, fs={fs}, order={order}")
                
                if cutoff <= 0 or fs <= 0 or order < 1:
                    raise ValueError('参数需为正数且阶数大于0')
                return {'cutoff': cutoff, 'fs': fs, 'order': order}
                
            elif filter_type == 'savitzky_golay':
                logger.info(f"处理Savitzky-Golay滤波参数")
                if not all(hasattr(self, attr) for attr in ['sg_window', 'sg_order']):
                    raise ValueError('Savitzky-Golay滤波参数控件未初始化')
                    
                window_size = int(self.sg_window.text())
                order = int(self.sg_order.text())
                logger.info(f"获取到参数: window_size={window_size}, order={order}")
                
                if window_size < 3 or order < 1 or window_size <= order:
                    raise ValueError('窗口需大于阶数且不少于3')
                return {'window_size': window_size, 'order': order}
                
            elif filter_type == 'threshold_filter':
                logger.info(f"处理限幅滤波参数，控件是否存在: {hasattr(self, 'threshold_value')}")
                if not hasattr(self, 'threshold_value'):
                    raise ValueError('限幅滤波阈值控件未初始化')
                    
                threshold = float(self.threshold_value.text())
                logger.info(f"获取到阈值参数: {threshold}")
                
                return {'threshold': threshold}
                
            elif filter_type == 'rate_limit_filter':
                logger.info(f"处理增幅限制滤波参数，控件是否存在: {hasattr(self, 'max_rate')}")
                if not hasattr(self, 'max_rate'):
                    raise ValueError('增幅限制滤波率控件未初始化')
                    
                max_rate = float(self.max_rate.text())
                logger.info(f"获取到最大变化率参数: {max_rate}")
                
                return {'max_rate': max_rate}
                
            elif filter_type == 'group_average_filter':
                logger.info(f"处理分组平均滤波参数，控件是否存在: {hasattr(self, 'group_size')}")
                if not hasattr(self, 'group_size'):
                    raise ValueError('分组平均滤波组大小控件未初始化')
                    
                group_size = int(self.group_size.text())
                logger.info(f"获取到组大小参数: {group_size}")
                
                if group_size < 2:
                    raise ValueError('组大小需大于1')
                return {'group_size': group_size}
                
            elif filter_type == 'custom':
                logger.info(f"处理自定义滤波参数，控件是否存在: {hasattr(self, 'formula_edit')}")
                if not hasattr(self, 'formula_edit'):
                    raise ValueError('自定义公式控件未初始化')
                    
                formula = self.formula_edit.toPlainText()
                logger.info(f"获取到公式: {formula}")
                
                if not formula.strip():
                    raise ValueError('自定义公式不能为空')
                return {'formula': formula}
                
            else:
                logger.info(f"未知滤波器类型: {filter_type}")
                return None
                
        except Exception as e:
            logger.error(f"获取滤波器参数时出错: {e}")
            logger.error(traceback.format_exc())
            QMessageBox.warning(self, "警告", f"获取滤波器参数时出错:\n{str(e)}")
            return None

    def update_filter_params(self):
        """更新滤波器参数显示"""
        try:
            logger.info(f"更新滤波器参数显示: 当前选择的滤波器为 {self.filter_combo.currentText()}")
            filter_type = self.filter_combo.currentText()
            
            # 隐藏所有参数控件
            for widget in self.param_widgets.values():
                widget.hide()
                
            # 显示当前滤波器的参数控件
            if filter_type in self.param_widgets:
                self.param_widgets[filter_type].show()
                logger.info(f"显示 {filter_type} 的参数控件")
        except Exception as e:
            logger.error(f"更新滤波器参数显示错误: {e}")
            logger.error(traceback.format_exc())
    
    def apply_filter(self, data):
        """应用滤波器"""
        filter_type = self.filter_combo.currentText()
        
        try:
            if filter_type == 'raw':
                return data
            elif filter_type == 'moving_average':
                window_size = int(self.ma_window.text())
                return TouchFilter.moving_average(data, window_size)
            elif filter_type == 'median_filter':
                window_size = int(self.median_window.text())
                return TouchFilter.median_filter(data, window_size)
            elif filter_type == 'exponential_moving_average':
                alpha = float(self.ema_alpha.text())
                return TouchFilter.exponential_moving_average(data, alpha)
            elif filter_type == 'butterworth_filter':
                cutoff = float(self.butter_cutoff.text())
                fs = float(self.butter_fs.text())
                order = int(self.butter_order.text())
                return TouchFilter.butterworth_filter(data, cutoff, fs, order)
            elif filter_type == 'savitzky_golay':
                window_size = int(self.sg_window.text())
                order = int(self.sg_order.text())
                return TouchFilter.savitzky_golay(data, window_size, order)
            elif filter_type == 'threshold_filter':
                threshold = float(self.threshold_value.text())
                return TouchFilter.threshold_filter(data, threshold)
            elif filter_type == 'rate_limit_filter':
                max_rate = float(self.max_rate.text())
                return TouchFilter.rate_limit_filter(data, max_rate)
            elif filter_type == 'group_average_filter':
                group_size = int(self.group_size.text())
                return TouchFilter.group_average_filter(data, group_size)
            elif filter_type == 'custom':
                return self.waveform_data.custom_filter.apply_filter(data)
            else:
                return data
        except Exception as e:
            logger.error(f"滤波器应用错误: {e}")
            QMessageBox.warning(self, "警告", f"应用滤波器时出错:\n{str(e)}")
            return data

    def load_csv(self):
        try:
            print("选择文件...")  # 调试信息
            filename, _ = QFileDialog.getOpenFileName(self, "选择CSV文件", "", "CSV files (*.csv)")
            if filename:
                print(f"加载文件: {filename}")  # 调试信息
                self.waveform_data.raw_data = pd.read_csv(filename)
                
                # 清空并更新波形列表
                self.waveform_list.clear()
                self.status_list.clear()
                self.style_list.clear()
                
                # 分离状态数据和波形数据
                for column in self.waveform_data.raw_data.columns:
                    if 'status' in column.lower():
                        self.status_list.addItem(column)
                    else:
                        self.waveform_list.addItem(column)
                        # 初始化波形样式
                        if column not in self.waveform_data.styles:
                            self.waveform_data.styles[column] = WaveformStyle()
                
                # 更新样式列表
                self.update_style_list()
                
                self.update_plot()
                print("文件加载完成")  # 调试信息
        except Exception as e:
            print(f"错误: {str(e)}")  # 调试信息
            QMessageBox.critical(self, "错误", f"加载文件时出错:\n{str(e)}")
                
    def update_plot(self):
        """更新图表显示"""
        print("更新图表...")  # 调试信息
        if self.waveform_data.raw_data is None:
            return
            
        try:
            self.ax.clear()
            self.ax2.clear()
            
            # 获取选中的波形
            selected_items = self.waveform_list.selectedItems()
            if not selected_items:
                return
                
            # 获取选中的状态数据
            status_item = self.status_list.selectedItems()
            status_data = None
            if status_item:
                status_column = status_item[0].text()
                status_data = self.waveform_data.raw_data[status_column].to_numpy()
            
            # 为每个选中的波形设置颜色
            filtered_applied = False  # 标记是否已经应用过滤波
            
            for i, item in enumerate(selected_items):
                column = item.text()
                
                # 使用波形样式
                if column not in self.waveform_data.styles:
                    self.waveform_data.styles[column] = WaveformStyle()
                
                style = self.waveform_data.styles[column]
                
                raw_data = self.waveform_data.raw_data[column].to_numpy()
                
                # 根据是否高亮设置线条粗细
                line_width = style.line_width * 1.5 if style.highlighted else style.line_width
                
                # 绘制原始数据，使用自定义样式
                self.ax.plot(raw_data, 
                           label=f'{column} (原始)',
                           color=style.color,
                           linestyle=style.line_style,
                           linewidth=line_width)
                
                # 应用级联滤波器，只应用于第一个选中的波形
                if self.waveform_data.active_filters and not filtered_applied and i == 0:
                    filtered_applied = True
                    filtered_data = self.waveform_data.apply_cascade_filters(raw_data)
                    # 使用更明显的对比：虚线、加粗线条、同色系但更深的颜色
                    filtered_color = self.darken_color(style.color)
                    filtered_line_width = line_width * 1.2
                    self.ax.plot(filtered_data, 
                               label=f'{column} (滤波后)',
                               color=filtered_color,
                               linewidth=filtered_line_width,
                               linestyle='--')
            
            # 绘制状态数据
            if status_data is not None:
                # 在第二个Y轴上绘制状态数据
                self.ax2.plot(status_data, 
                            label='状态',
                            color='gray',
                            linewidth=1.0,
                            alpha=0.5)
                # 设置第二个Y轴的范围为0-1
                self.ax2.set_ylim(-0.1, 1.1)
                self.ax2.set_ylabel('状态', fontsize=12)
            
            # 设置标题和标签
            self.ax.set_title('波形显示', fontsize=14)
            self.ax.set_xlabel('采样点', fontsize=12)
            self.ax.set_ylabel('数值', fontsize=12)
            
            # 添加图例
            lines1, labels1 = self.ax.get_legend_handles_labels()
            lines2, labels2 = self.ax2.get_legend_handles_labels()
            self.ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
            
            # 应用坐标范围（如果已设置）
            if self.x_min.text() and self.x_max.text():
                try:
                    x_min = float(self.x_min.text())
                    x_max = float(self.x_max.text())
                    if x_min < x_max:
                        self.ax.set_xlim(x_min, x_max)
                except ValueError:
                    pass
            else:
                # 自动调整坐标范围
                self.ax.relim()
                self.ax.autoscale_view()
                
            if self.y_min.text() and self.y_max.text():
                try:
                    y_min = float(self.y_min.text())
                    y_max = float(self.y_max.text())
                    if y_min < y_max:
                        self.ax.set_ylim(y_min, y_max)
                except ValueError:
                    pass
            
            self.figure.tight_layout()
            self.canvas.draw()
            print("图表更新完成")  # 调试信息
        except Exception as e:
            print(f"绘图错误: {str(e)}")  # 调试信息
            QMessageBox.warning(self, "警告", f"更新图表时出错:\n{str(e)}")
            
    def darken_color(self, color):
        """使颜色更深，增强对比度"""
        if color == 'b':  # 蓝色
            return 'navy'
        elif color == 'g':  # 绿色
            return 'darkgreen'
        elif color == 'r':  # 红色
            return 'darkred'
        elif color == 'c':  # 青色
            return 'darkcyan'
        elif color == 'm':  # 洋红
            return 'darkmagenta'
        elif color == 'y':  # 黄色
            return 'goldenrod'
        elif color == 'k':  # 黑色
            return 'dimgray'
        else:
            return color

    def apply_axis_range(self):
        """应用用户设定的坐标轴范围"""
        try:
            logger.info("应用坐标轴范围")
            
            # 获取X轴范围
            if self.x_min.text() and self.x_max.text():
                try:
                    x_min = float(self.x_min.text())
                    x_max = float(self.x_max.text())
                    if x_min < x_max:
                        self.ax.set_xlim(x_min, x_max)
                        logger.info(f"设置X轴范围: {x_min} 到 {x_max}")
                    else:
                        logger.warning("X轴最小值应小于最大值")
                except ValueError:
                    logger.warning("X轴范围必须是数值")
            
            # 获取Y轴范围
            if self.y_min.text() and self.y_max.text():
                try:
                    y_min = float(self.y_min.text())
                    y_max = float(self.y_max.text())
                    if y_min < y_max:
                        self.ax.set_ylim(y_min, y_max)
                        logger.info(f"设置Y轴范围: {y_min} 到 {y_max}")
                    else:
                        logger.warning("Y轴最小值应小于最大值")
                except ValueError:
                    logger.warning("Y轴范围必须是数值")
            
            # 刷新画布
            self.canvas.draw()
            logger.info("坐标轴范围已应用")
        except Exception as e:
            logger.error(f"应用坐标轴范围错误: {e}")
            logger.error(traceback.format_exc())

    def show_style_menu(self, position):
        """显示波形样式上下文菜单"""
        try:
            menu = QMenu()
            item = self.style_list.itemAt(position)
            if item:
                # 设置颜色
                color_action = menu.addAction("设置颜色")
                color_action.triggered.connect(lambda: self.set_waveform_color(item))
                
                # 设置线型
                line_style_menu = menu.addMenu("设置线型")
                
                solid_action = line_style_menu.addAction("实线")
                solid_action.triggered.connect(lambda: self.set_waveform_line_style(item, '-'))
                
                dashed_action = line_style_menu.addAction("虚线")
                dashed_action.triggered.connect(lambda: self.set_waveform_line_style(item, '--'))
                
                dotted_action = line_style_menu.addAction("点线")
                dotted_action.triggered.connect(lambda: self.set_waveform_line_style(item, ':'))
                
                dash_dot_action = line_style_menu.addAction("点划线")
                dash_dot_action.triggered.connect(lambda: self.set_waveform_line_style(item, '-.'))
                
                # 设置线宽
                line_width_menu = menu.addMenu("设置线宽")
                
                thin_action = line_width_menu.addAction("细")
                thin_action.triggered.connect(lambda: self.set_waveform_line_width(item, 1.0))
                
                normal_action = line_width_menu.addAction("中")
                normal_action.triggered.connect(lambda: self.set_waveform_line_width(item, 1.5))
                
                thick_action = line_width_menu.addAction("粗")
                thick_action.triggered.connect(lambda: self.set_waveform_line_width(item, 2.0))
                
                very_thick_action = line_width_menu.addAction("极粗")
                very_thick_action.triggered.connect(lambda: self.set_waveform_line_width(item, 3.0))
                
                # 高亮设置
                highlight_action = menu.addAction("高亮波形")
                highlight_action.triggered.connect(lambda: self.toggle_highlight(item))
                
            menu.exec_(self.style_list.mapToGlobal(position))
        except Exception as e:
            logger.error(f"显示样式菜单错误: {e}")
            logger.error(traceback.format_exc())
    
    def set_waveform_color(self, item):
        """设置波形颜色"""
        try:
            column = item.text()
            if column in self.waveform_data.styles:
                current_color = QColor(self.waveform_data.styles[column].color)
                color = QColorDialog.getColor(current_color, self, "选择波形颜色")
                if color.isValid():
                    self.waveform_data.styles[column].color = color.name()
                    self.update_style_list()
                    self.update_plot()
        except Exception as e:
            logger.error(f"设置波形颜色错误: {e}")
            logger.error(traceback.format_exc())
    
    def set_waveform_line_style(self, item, style):
        """设置波形线型"""
        try:
            column = item.text()
            if column in self.waveform_data.styles:
                self.waveform_data.styles[column].line_style = style
                self.update_style_list()
                self.update_plot()
        except Exception as e:
            logger.error(f"设置波形线型错误: {e}")
            logger.error(traceback.format_exc())
    
    def set_waveform_line_width(self, item, width):
        """设置波形线宽"""
        try:
            column = item.text()
            if column in self.waveform_data.styles:
                self.waveform_data.styles[column].line_width = width
                self.update_style_list()
                self.update_plot()
        except Exception as e:
            logger.error(f"设置波形线宽错误: {e}")
            logger.error(traceback.format_exc())
    
    def toggle_highlight(self, item):
        """切换波形的高亮状态"""
        try:
            column = item.text()
            if column in self.waveform_data.styles:
                self.waveform_data.styles[column].highlighted = not self.waveform_data.styles[column].highlighted
                self.update_style_list()
                self.update_plot()
        except Exception as e:
            logger.error(f"切换高亮状态错误: {e}")
            logger.error(traceback.format_exc())
    
    def highlight_selected_waveform(self):
        """高亮选中的波形"""
        try:
            selected_items = self.style_list.selectedItems()
            if not selected_items:
                return
                
            # 先将所有波形设为非高亮
            for column in self.waveform_data.styles:
                self.waveform_data.styles[column].highlighted = False
                
            # 将选中的波形设为高亮
            for item in selected_items:
                column = item.text()
                if column in self.waveform_data.styles:
                    self.waveform_data.styles[column].highlighted = True
                    
            self.update_style_list()
            self.update_plot()
        except Exception as e:
            logger.error(f"高亮选中波形错误: {e}")
            logger.error(traceback.format_exc())
    
    def update_style_list(self):
        """更新样式列表的显示"""
        try:
            self.style_list.clear()
            for column, style in self.waveform_data.styles.items():
                item = QListWidgetItem(column)
                if style.highlighted:
                    item.setBackground(QColor(240, 240, 150))  # 高亮背景色
                self.style_list.addItem(item)
        except Exception as e:
            logger.error(f"更新样式列表错误: {e}")
            logger.error(traceback.format_exc())
    
    def update_filter_code(self):
        """更新滤波器代码显示"""
        try:
            if not self.waveform_data.active_filters:
                self.filter_code_text.setText("未添加滤波器")
                return
                
            language = self.code_language.currentText()
            code_text = ""
            
            for i, (filter_type, params) in enumerate(self.waveform_data.active_filters):
                code = self.waveform_data.get_filter_code(filter_type, params, language)
                code_text += f"# 滤波器 {i+1}: {filter_type}\n{code}\n\n"
                
            self.filter_code_text.setText(code_text)
        except Exception as e:
            logger.error(f"更新滤波器代码错误: {e}")
            logger.error(traceback.format_exc())

def main():
    try:
        logger.info("启动应用程序...")
        app = QApplication(sys.argv)
        window = TouchFilterGUI()
        window.show()
        logger.info("窗口已显示")
        return app.exec_()
    except Exception as e:
        logger.error(f"程序错误: {e}")
        logger.error(traceback.format_exc())
        QMessageBox.critical(None, "严重错误", f"程序运行时出错:\n{str(e)}")
        return 1

if __name__ == '__main__':
    logger.info("程序开始")
    sys.exit(main()) 