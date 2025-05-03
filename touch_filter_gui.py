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
                               QColorDialog, QListWidgetItem, QMenu, QAction, QMenuBar,
                               QToolBar, QStatusBar, QDockWidget, QDialog,
                               QGridLayout)
    from PyQt5.QtCore import Qt, QPointF
    from PyQt5.QtGui import QFont, QColor, QCursor, QIntValidator, QDoubleValidator
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
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
            
            # 保存坐标范围
            self.current_x_range = None
            self.current_y_range = None
            self.auto_range = True  # 是否使用自动范围
            
            # 创建主菜单
            self._create_menus()
            
            # 创建状态栏
            self.statusBar = QStatusBar()
            self.setStatusBar(self.statusBar)
            
            # 创建主窗口布局
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            main_layout = QVBoxLayout(central_widget)
            
            # 滤波器参数控件字典初始化
            self.filter_param_widgets = {}
            
            # 创建图表区域
            self._create_plot_panel(main_layout)
            
            # 创建滤波器设置面板
            self._create_filter_panel(main_layout)
            
            # 波形点击选择相关变量初始化
            self.selected_line = None
            self.selected_line_index = -1
            self.selected_curve_name = None
            
            # 初始化内部数据列表（用于存储选中状态，但不显示在界面上）
            self.waveform_list = QListWidget()
            self.waveform_list.setSelectionMode(QListWidget.MultiSelection)
            self.waveform_list.hide()  # 隐藏，不显示在界面上
            
            # 初始化波形样式列表
            self.style_list = QListWidget()
            self.style_list.hide()  # 隐藏，不显示在界面上
            
            # 显示欢迎信息
            self.statusBar.showMessage("欢迎使用触摸数据滤波分析工具，请加载CSV文件开始分析")
            
            logger.info("GUI初始化完成")
        except Exception as e:
            logger.error(f"GUI初始化错误: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _create_plot_panel(self, parent_layout):
        """创建图表面板"""
        try:
            logger.info("创建图表面板")
            
            # 创建图表
            self.figure = Figure(figsize=(8, 6))
            self.canvas = FigureCanvas(self.figure)
            self.ax = self.figure.add_subplot(111)
            self.ax2 = self.ax.twinx()  # 创建第二个Y轴（保留用于兼容性）
            
            # 设置图表标题和标签
            self.ax.set_title('波形显示', fontsize=14)
            self.ax.set_xlabel('采样点', fontsize=12)
            self.ax.set_ylabel('数值', fontsize=12)
            
            # 添加点击事件处理
            self.canvas.mpl_connect('pick_event', self.on_pick)
            
            # 添加图表到布局
            parent_layout.addWidget(self.canvas)
            
            logger.info("图表面板创建完成")
        except Exception as e:
            logger.error(f"创建图表面板错误: {e}")
            logger.error(traceback.format_exc())
    
    def _create_waveform_selection_panel(self, parent_layout):
        """创建波形选择面板"""
        try:
            logger.info("创建波形选择面板")
            
            # 创建波形列表组
            waveform_group = QGroupBox("波形选择")
            waveform_layout = QVBoxLayout()
            
            # 添加多选列表
            self.waveform_list = QListWidget()
            self.waveform_list.setSelectionMode(QListWidget.MultiSelection)
            self.waveform_list.itemSelectionChanged.connect(self.update_plot)
            waveform_layout.addWidget(self.waveform_list)
            waveform_group.setLayout(waveform_layout)
            
            # 添加到主布局
            parent_layout.addWidget(waveform_group)
            
            logger.info("波形选择面板创建完成")
        except Exception as e:
            logger.error(f"创建波形选择面板错误: {e}")
            logger.error(traceback.format_exc())
    
    def _create_filter_param_widget(self, filter_type):
        """创建滤波器参数控件"""
        try:
            widget = QWidget()
            layout = QVBoxLayout(widget)
            
            if filter_type == 'moving_average':
                # 移动平均滤波参数
                param_layout = QHBoxLayout()
                param_layout.addWidget(QLabel("窗口大小:"))
                self.ma_window = QLineEdit("5")
                self.ma_window.setValidator(QIntValidator(1, 1000))
                param_layout.addWidget(self.ma_window)
                layout.addLayout(param_layout)
                
            elif filter_type == 'median_filter':
                # 中值滤波参数
                param_layout = QHBoxLayout()
                param_layout.addWidget(QLabel("窗口大小:"))
                self.median_window = QLineEdit("5")
                self.median_window.setValidator(QIntValidator(1, 1000))
                param_layout.addWidget(self.median_window)
                layout.addLayout(param_layout)
                
            elif filter_type == 'exponential_moving_average':
                # 指数移动平均参数
                param_layout = QHBoxLayout()
                param_layout.addWidget(QLabel("Alpha值:"))
                self.ema_alpha = QLineEdit("0.1")
                self.ema_alpha.setValidator(QDoubleValidator(0.0, 1.0, 3))
                param_layout.addWidget(self.ema_alpha)
                layout.addLayout(param_layout)
                
            elif filter_type == 'butterworth_filter':
                # 巴特沃斯滤波参数
                param_layout = QGridLayout()
                param_layout.addWidget(QLabel("截止频率:"), 0, 0)
                self.butter_cutoff = QLineEdit("5.0")
                self.butter_cutoff.setValidator(QDoubleValidator(0.01, 1000.0, 2))
                param_layout.addWidget(self.butter_cutoff, 0, 1)
                param_layout.addWidget(QLabel("采样频率:"), 1, 0)
                self.butter_fs = QLineEdit("60.0")
                self.butter_fs.setValidator(QDoubleValidator(0.01, 10000.0, 2))
                param_layout.addWidget(self.butter_fs, 1, 1)
                param_layout.addWidget(QLabel("阶数:"), 2, 0)
                self.butter_order = QLineEdit("3")
                self.butter_order.setValidator(QIntValidator(1, 20))
                param_layout.addWidget(self.butter_order, 2, 1)
                layout.addLayout(param_layout)
                
            elif filter_type == 'savitzky_golay':
                # Savitzky-Golay滤波参数
                param_layout = QGridLayout()
                param_layout.addWidget(QLabel("窗口大小:"), 0, 0)
                self.sg_window = QLineEdit("5")
                self.sg_window.setValidator(QIntValidator(3, 1001))
                param_layout.addWidget(self.sg_window, 0, 1)
                param_layout.addWidget(QLabel("阶数:"), 1, 0)
                self.sg_order = QLineEdit("2")
                self.sg_order.setValidator(QIntValidator(1, 10))
                param_layout.addWidget(self.sg_order, 1, 1)
                layout.addLayout(param_layout)
                
            elif filter_type == 'threshold_filter':
                # 限幅滤波参数
                param_layout = QHBoxLayout()
                param_layout.addWidget(QLabel("阈值:"))
                self.threshold_value = QLineEdit("100")
                self.threshold_value.setValidator(QDoubleValidator(-1000000.0, 1000000.0, 2))
                param_layout.addWidget(self.threshold_value)
                layout.addLayout(param_layout)
                
            elif filter_type == 'rate_limit_filter':
                # 增幅限制滤波参数
                param_layout = QHBoxLayout()
                param_layout.addWidget(QLabel("最大变化率:"))
                self.max_rate = QLineEdit("10")
                self.max_rate.setValidator(QDoubleValidator(0.0, 1000000.0, 2))
                param_layout.addWidget(self.max_rate)
                layout.addLayout(param_layout)
                
            elif filter_type == 'group_average_filter':
                # 分组平均滤波参数
                param_layout = QHBoxLayout()
                param_layout.addWidget(QLabel("组大小:"))
                self.group_size = QLineEdit("6")
                self.group_size.setValidator(QIntValidator(2, 1000))
                param_layout.addWidget(self.group_size)
                layout.addLayout(param_layout)
                
            elif filter_type == 'custom':
                # 自定义公式滤波参数
                self.formula_edit = QTextEdit()
                self.formula_edit.setPlaceholderText("输入Python表达式，例如：\n"
                                                 "np.convolve(data, np.ones(5)/5, 'same')")
                self.formula_edit.setMaximumHeight(60)
                layout.addWidget(QLabel("公式(使用data表示输入数据):"))
                layout.addWidget(self.formula_edit)
            
            return widget
        except Exception as e:
            logger.error(f"创建滤波器参数控件时出错: {e}")
            logger.error(traceback.format_exc())
            return QWidget()  # 返回空控件，避免程序崩溃
    
    def _initialize_filter_params(self):
        """初始化滤波器参数字典，不再单独创建控件，而是通过_create_filter_param_widget方法创建"""
        try:
            logger.info("初始化滤波器参数字典")
            self.filter_param_widgets = {}
            logger.info("滤波器参数字典初始化完成")
        except Exception as e:
            logger.error(f"初始化滤波器参数字典时出错: {e}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "错误", f"初始化滤波器参数字典时出错:\n{str(e)}")
    
    def toggle_filter_params(self, filter_type, checked):
        """切换滤波器参数面板的显示状态"""
        try:
            logger.info(f"切换'{filter_type}'滤波器参数控件显示: {'显示' if checked else '隐藏'}")
            
            if filter_type in self.filter_param_widgets:
                for widget in self.filter_param_widgets[filter_type]:
                    if checked:
                        widget.show()
                    else:
                        widget.hide()
            
            # 更新界面布局
            if hasattr(self, 'filter_scroll_area'):
                self.filter_scroll_area.updateGeometry()
                
            # 记录选中状态
            if hasattr(self, 'filter_checkboxes') and filter_type in self.filter_checkboxes:
                self.filter_checkboxes[filter_type].setChecked(checked)
        except Exception as e:
            logger.error(f"切换滤波器参数控件显示状态时出错: {e}")
            logger.error(traceback.format_exc())
    
    def apply_selected_filters(self):
        """应用所有选中的滤波器，为每个滤波器创建新的数据通道"""
        try:
            logger.info("应用选中的滤波器")
            
            # 获取选中的滤波器
            active_filters = []
            has_checked = False
            
            # 从菜单和复选框获取选中的滤波器
            for filter_type, action in self.filter_menu_actions.items():
                if action.isChecked():
                    has_checked = True
                    logger.info(f"获取滤波器 {filter_type} 的参数")
                    params = self.get_filter_params(filter_type)
                    if params:
                        logger.info(f"添加滤波器 {filter_type} 到级联列表")
                        active_filters.append((filter_type, params))
            
            # 如果没有选中的滤波器，提示用户
            if not has_checked:
                QMessageBox.information(self, "提示", "请选择至少一个滤波器")
                return
                
            # 如果获取参数失败，可能会导致active_filters为空
            if not active_filters:
                QMessageBox.warning(self, "警告", "无法获取滤波器参数，请检查参数设置")
                return
            
            # 保存当前选中的波形
            selected_waveforms = self.get_selected_waveforms()
            if not selected_waveforms:
                QMessageBox.information(self, "提示", "请选择至少一个波形")
                return
                
            # 保存当前坐标范围
            try:
                if hasattr(self.ax, 'get_xlim') and hasattr(self.ax, 'get_ylim'):
                    self.current_x_range = self.ax.get_xlim()
                    self.current_y_range = self.ax.get_ylim()
                    logger.info(f"保存当前坐标范围: X={self.current_x_range}, Y={self.current_y_range}")
            except Exception as range_e:
                logger.error(f"保存坐标范围错误: {range_e}")
            
            # 为每个选中的波形应用所有滤波器并创建新的数据通道
            new_columns_added = []
            
            for column in selected_waveforms:
                logger.info(f"处理波形: {column}")
                
                # 获取原始数据
                try:
                    raw_data = self.waveform_data.raw_data[column].values
                except Exception as e:
                    logger.error(f"获取波形数据错误: {e}")
                    continue
                
                # 依次应用每个滤波器
                for filter_type, params in active_filters:
                    try:
                        # 获取滤波器方法
                        filter_method = getattr(TouchFilter, filter_type, None)
                        if filter_method:
                            logger.info(f"应用滤波器 {filter_type} 到波形 {column}")
                            
                            # 应用滤波器
                            filtered_data = filter_method(raw_data.copy(), **params)
                            
                            # 创建新的数据通道名
                            filter_display_name = self._get_filter_display_name(filter_type)
                            new_column_name = f"{column}_{filter_display_name}"
                            
                            # 处理名称冲突
                            suffix = 1
                            original_name = new_column_name
                            while new_column_name in self.waveform_data.raw_data.columns:
                                new_column_name = f"{original_name}_{suffix}"
                                suffix += 1
                            
                            # 添加滤波后的数据到数据集
                            self.waveform_data.raw_data[new_column_name] = filtered_data
                            
                            # 为新通道创建样式（继承原始波形样式，但颜色深一点）
                            if column in self.waveform_data.styles:
                                original_style = self.waveform_data.styles[column]
                                filtered_style = WaveformStyle(
                                    color=self.darken_color(original_style.color),
                                    line_style=original_style.line_style,
                                    line_width=original_style.line_width,
                                    highlighted=original_style.highlighted
                                )
                                self.waveform_data.styles[new_column_name] = filtered_style
                            
                            # 添加到波形列表
                            item = QListWidgetItem(new_column_name)
                            self.waveform_list.addItem(item)
                            item.setSelected(True)  # 自动选中新添加的波形
                            
                            # 添加到波形选择菜单
                            action = QAction(new_column_name, self)
                            action.setCheckable(True)
                            action.setChecked(True)  # 默认选中新创建的通道
                            action.triggered.connect(lambda checked, col=new_column_name: self.toggle_waveform_selection(col, checked))
                            self.waveform_selection_menu.addAction(action)
                            self.waveform_actions[new_column_name] = action
                            
                            # 记录新创建的列
                            new_columns_added.append(new_column_name)
                            
                            logger.info(f"已创建新数据通道: {new_column_name}")
                        else:
                            logger.warning(f"找不到滤波器方法: {filter_type}")
                    except Exception as filter_e:
                        logger.error(f"应用滤波器 {filter_type} 时出错: {filter_e}")
                        logger.error(traceback.format_exc())
            
            # 更新样式列表
            if hasattr(self, 'update_style_list'):
                self.update_style_list()
            
            # 更新波形菜单的选中状态
            for column, action in self.waveform_actions.items():
                action.setChecked(column in new_columns_added or column in selected_waveforms)
            
            # 更新图表
            self.update_plot()
            
            # 提示成功
            if new_columns_added:
                self.statusBar.showMessage(f"已创建 {len(new_columns_added)} 个新数据通道")
            else:
                self.statusBar.showMessage("应用滤波器完成，但未创建新通道")
                
            logger.info("应用滤波器完成")
        except Exception as e:
            logger.error(f"应用滤波器错误: {e}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "错误", f"应用滤波器时发生错误：\n{str(e)}")
    
    def clear_all_filters(self):
        """清除所有滤波器"""
        try:
            logger.info("清除所有滤波器")
            
            # 清除滤波器列表
            self.waveform_data.active_filters = []
            
            # 取消所有菜单项的选中状态
            for action in self.filter_menu_actions.values():
                action.setChecked(False)
            
            # 取消所有复选框选中状态
            for checkbox in self.filter_checkboxes.values():
                checkbox.setChecked(False)
            
            # 隐藏所有参数面板
            for widgets in self.filter_param_widgets.values():
                for widget in widgets:
                    widget.hide()
            
            # 更新图表
            self.update_plot()
            
            self.statusBar.showMessage("已清除所有滤波器")
            logger.info("已清除所有滤波器")
        except Exception as e:
            logger.error(f"清除滤波器错误: {e}")
            logger.error(traceback.format_exc())
    
    def get_filter_params(self, filter_type):
        """获取滤波器参数，增加健壮性校验，防止闪退"""
        try:
            logger.info(f"获取'{filter_type}'滤波器参数")
            
            if filter_type == 'moving_average':
                if not hasattr(self, 'ma_window'):
                    raise ValueError('移动平均窗口控件未初始化')
                    
                val = self.ma_window.text().strip()
                logger.info(f"获取到窗口大小参数: {val}")
                
                if not val or not val.isdigit() or int(val) < 1:
                    QMessageBox.warning(self, "参数错误", "移动平均窗口大小需为正整数")
                    return None
                return {'window_size': int(val)}
                
            elif filter_type == 'median_filter':
                if not hasattr(self, 'median_window'):
                    raise ValueError('中值滤波窗口控件未初始化')
                    
                val = self.median_window.text().strip()
                logger.info(f"获取到窗口大小参数: {val}")
                
                if not val or not val.isdigit() or int(val) < 1:
                    QMessageBox.warning(self, "参数错误", "中值滤波窗口大小需为正整数")
                    return None
                return {'window_size': int(val)}
                
            elif filter_type == 'exponential_moving_average':
                if not hasattr(self, 'ema_alpha'):
                    raise ValueError('指数移动平均Alpha控件未初始化')
                    
                val = self.ema_alpha.text().strip()
                logger.info(f"获取到Alpha参数: {val}")
                
                try:
                    alpha = float(val)
                    if not (0 < alpha < 1):
                        QMessageBox.warning(self, "参数错误", "Alpha值需在0和1之间")
                        return None
                    return {'alpha': alpha}
                except ValueError:
                    QMessageBox.warning(self, "参数错误", "Alpha值需为有效的小数")
                    return None
                
            elif filter_type == 'butterworth_filter':
                if not all(hasattr(self, attr) for attr in ['butter_cutoff', 'butter_fs', 'butter_order']):
                    raise ValueError('巴特沃斯滤波参数控件未初始化')
                    
                try:
                    cutoff = float(self.butter_cutoff.text().strip())
                    fs = float(self.butter_fs.text().strip())
                    order = int(self.butter_order.text().strip())
                    logger.info(f"获取到参数: cutoff={cutoff}, fs={fs}, order={order}")
                    
                    if cutoff <= 0 or fs <= 0 or order < 1:
                        QMessageBox.warning(self, "参数错误", "所有参数需为正数且阶数大于0")
                        return None
                    return {'cutoff': cutoff, 'fs': fs, 'order': order}
                except ValueError:
                    QMessageBox.warning(self, "参数错误", "请输入有效的数值")
                    return None
                
            elif filter_type == 'savitzky_golay':
                if not all(hasattr(self, attr) for attr in ['sg_window', 'sg_order']):
                    raise ValueError('Savitzky-Golay滤波参数控件未初始化')
                
                try:
                    window_size = int(self.sg_window.text().strip())
                    order = int(self.sg_order.text().strip())
                    logger.info(f"获取到参数: window_size={window_size}, order={order}")
                    
                    if window_size < 3 or order < 1 or window_size <= order:
                        QMessageBox.warning(self, "参数错误", "窗口需大于阶数且不少于3")
                        return None
                    return {'window_size': window_size, 'order': order}
                except ValueError:
                    QMessageBox.warning(self, "参数错误", "请输入有效的整数")
                    return None
                
            elif filter_type == 'threshold_filter':
                if not hasattr(self, 'threshold_value'):
                    raise ValueError('限幅滤波阈值控件未初始化')
                
                try:
                    threshold = float(self.threshold_value.text().strip())
                    logger.info(f"获取到阈值参数: {threshold}")
                    return {'threshold': threshold}
                except ValueError:
                    QMessageBox.warning(self, "参数错误", "阈值需为有效的数值")
                    return None
                
            elif filter_type == 'rate_limit_filter':
                if not hasattr(self, 'max_rate'):
                    raise ValueError('增幅限制滤波率控件未初始化')
                
                try:
                    max_rate = float(self.max_rate.text().strip())
                    logger.info(f"获取到最大变化率参数: {max_rate}")
                    return {'max_rate': max_rate}
                except ValueError:
                    QMessageBox.warning(self, "参数错误", "最大变化率需为有效的数值")
                    return None
                
            elif filter_type == 'group_average_filter':
                if not hasattr(self, 'group_size'):
                    raise ValueError('分组平均滤波组大小控件未初始化')
                
                try:
                    group_size = int(self.group_size.text().strip())
                    logger.info(f"获取到组大小参数: {group_size}")
                    
                    if group_size < 2:
                        QMessageBox.warning(self, "参数错误", "组大小需大于1")
                        return None
                    return {'group_size': group_size}
                except ValueError:
                    QMessageBox.warning(self, "参数错误", "请输入有效的整数")
                    return None
                
            elif filter_type == 'custom':
                if not hasattr(self, 'formula_edit'):
                    raise ValueError('自定义公式控件未初始化')
                    
                formula = self.formula_edit.toPlainText().strip()
                logger.info(f"获取到公式: {formula}")
                
                if not formula:
                    QMessageBox.warning(self, "参数错误", "自定义公式不能为空")
                    return None
                return {'formula': formula}
                
            else:
                logger.info(f"未知滤波器类型: {filter_type}")
                return None
                
        except Exception as e:
            logger.error(f"获取滤波器参数时出错: {e}")
            logger.error(traceback.format_exc())
            QMessageBox.warning(self, "警告", f"获取滤波器参数时出错:\n{str(e)}")
            return None

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

    def on_pick(self, event):
        """处理点击选择波形事件"""
        try:
            # 确保点击的是线条
            if isinstance(event.artist, Line2D):
                # 获取点击的线条和索引
                thisline = event.artist
                self.selected_line = thisline
                self.selected_line_index = event.ind[0]
                
                # 获取点击位置
                xdata = thisline.get_xdata()
                ydata = thisline.get_ydata()
                
                # 获取波形名称
                label = thisline.get_label()
                if '(' in label:
                    # 分离出波形名称和类型（原始/滤波）
                    curve_name = label.split('(')[0].strip()
                    curve_type = label.split('(')[1].replace(')', '').strip()
                else:
                    curve_name = label
                    curve_type = "未知"
                
                self.selected_curve_name = curve_name
                
                # 在状态栏显示选中的波形和坐标
                if 0 <= self.selected_line_index < len(xdata):
                    self.statusBar.showMessage(f"选中波形: {curve_name} ({curve_type}), 位置: x={xdata[self.selected_line_index]:.2f}, y={ydata[self.selected_line_index]:.2f}")
                
                # 更新样式列表选择状态
                if hasattr(self, 'style_list'):
                    for i in range(self.style_list.count()):
                        item = self.style_list.item(i)
                        if item.text() == curve_name:
                            self.style_list.setCurrentItem(item)
                            break
                
                # 弹出样式编辑菜单
                self.show_waveform_menu(event.mouseevent)
                
                # 刷新图表
                self.canvas.draw_idle()
        except Exception as e:
            logger.error(f"波形选择错误: {e}")
            logger.error(traceback.format_exc())
    
    def show_waveform_menu(self, mouse_event):
        """显示波形样式编辑菜单"""
        try:
            if self.selected_line is None or self.selected_curve_name is None:
                return
                
            # 获取波形名称
            column = self.selected_curve_name
                
            # 确保波形存在样式
            if column not in self.waveform_data.styles:
                self.waveform_data.styles[column] = WaveformStyle()
                
            # 创建上下文菜单
            menu = QMenu(self)
            
            # 波形信息
            label = self.selected_line.get_label()
            info_action = menu.addAction(f"波形: {label}")
            info_action.setEnabled(False)
            menu.addSeparator()
            
            # 设置颜色
            color_action = menu.addAction("设置颜色")
            color_action.triggered.connect(lambda: self.set_waveform_color_from_menu(column))
            
            # 设置线型
            line_style_menu = menu.addMenu("设置线型")
            
            solid_action = line_style_menu.addAction("实线")
            solid_action.triggered.connect(lambda: self.set_waveform_line_style_from_menu(column, '-'))
            
            dashed_action = line_style_menu.addAction("虚线")
            dashed_action.triggered.connect(lambda: self.set_waveform_line_style_from_menu(column, '--'))
            
            dotted_action = line_style_menu.addAction("点线")
            dotted_action.triggered.connect(lambda: self.set_waveform_line_style_from_menu(column, ':'))
            
            dash_dot_action = line_style_menu.addAction("点划线")
            dash_dot_action.triggered.connect(lambda: self.set_waveform_line_style_from_menu(column, '-.'))
            
            # 设置线宽
            line_width_menu = menu.addMenu("设置线宽")
            
            thin_action = line_width_menu.addAction("细")
            thin_action.triggered.connect(lambda: self.set_waveform_line_width_from_menu(column, 1.0))
            
            normal_action = line_width_menu.addAction("中")
            normal_action.triggered.connect(lambda: self.set_waveform_line_width_from_menu(column, 1.5))
            
            thick_action = line_width_menu.addAction("粗")
            thick_action.triggered.connect(lambda: self.set_waveform_line_width_from_menu(column, 2.0))
            
            very_thick_action = line_width_menu.addAction("极粗")
            very_thick_action.triggered.connect(lambda: self.set_waveform_line_width_from_menu(column, 3.0))
            
            # 高亮设置
            highlight_action = menu.addAction("高亮波形")
            highlight_action.setCheckable(True)
            highlight_action.setChecked(self.waveform_data.styles[column].highlighted)
            highlight_action.triggered.connect(lambda: self.toggle_highlight_from_menu(column))
            
            # 高亮单独此波形（降低其他波形亮度）
            solo_highlight_action = menu.addAction("单独高亮此波形")
            solo_highlight_action.triggered.connect(lambda: self.solo_highlight_waveform(column))
            
            # 重置所有波形亮度
            reset_highlight_action = menu.addAction("重置所有高亮")
            reset_highlight_action.triggered.connect(self.reset_all_highlights)
            
            # 显示菜单
            cursor = QCursor.pos()
            menu.exec_(cursor)
        except Exception as e:
            logger.error(f"显示波形菜单错误: {e}")
            logger.error(traceback.format_exc())
    
    def set_waveform_color_from_menu(self, column):
        """从图表菜单设置波形颜色"""
        try:
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
    
    def set_waveform_line_style_from_menu(self, column, style):
        """从图表菜单设置波形线型"""
        try:
            if column in self.waveform_data.styles:
                self.waveform_data.styles[column].line_style = style
                self.update_style_list()
                self.update_plot()
        except Exception as e:
            logger.error(f"设置波形线型错误: {e}")
            logger.error(traceback.format_exc())
    
    def set_waveform_line_width_from_menu(self, column, width):
        """从图表菜单设置波形线宽"""
        try:
            if column in self.waveform_data.styles:
                self.waveform_data.styles[column].line_width = width
                self.update_style_list()
                self.update_plot()
        except Exception as e:
            logger.error(f"设置波形线宽错误: {e}")
            logger.error(traceback.format_exc())
    
    def toggle_highlight_from_menu(self, column):
        """从图表菜单切换高亮状态"""
        try:
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

    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(self, "关于触摸数据滤波分析工具", 
                         "触摸数据滤波分析工具\n\n"
                         "版本: 1.0\n"
                         "一个用于可视化和处理触摸数据的工具\n\n"
                         "支持多种滤波方法和波形自定义")

    def load_csv(self):
        """加载CSV文件"""
        try:
            logger.info("开始加载CSV文件")
            filename, _ = QFileDialog.getOpenFileName(self, "选择CSV文件", "", "CSV files (*.csv);;All files (*.*)")
            if filename:
                logger.info(f"选择文件: {filename}")
                
                # 加载数据
                self.waveform_data.raw_data = pd.read_csv(filename)
                logger.info(f"成功加载数据，列数: {len(self.waveform_data.raw_data.columns)}")
                
                # 清空波形选择菜单
                self.waveform_selection_menu.clear()
                
                # 存储波形菜单动作，用于管理选中状态
                self.waveform_actions = {}
                
                # 清空内部选择列表（仍然保留这些列表用于内部存储选中状态）
                self.waveform_list.clear()
                
                if hasattr(self, 'style_list'):
                    self.style_list.clear()
                
                # 添加所有列为波形数据
                for column in self.waveform_data.raw_data.columns:
                    # 添加到波形数据列表（内部存储）
                    self.waveform_list.addItem(column)
                    logger.info(f"添加波形数据: {column}")
                    
                    # 初始化波形样式
                    if column not in self.waveform_data.styles:
                        self.waveform_data.styles[column] = WaveformStyle()
                    
                    # 添加到波形选择菜单
                    action = QAction(column, self)
                    action.setCheckable(True)
                    action.triggered.connect(lambda checked, col=column: self.toggle_waveform_selection(col, checked))
                    self.waveform_selection_menu.addAction(action)
                    self.waveform_actions[column] = action
                
                # 更新样式列表
                if hasattr(self, 'update_style_list'):
                    self.update_style_list()
                
                # 更新图表
                self.update_plot()
                
                # 更新窗口标题
                self.setWindowTitle(f'触摸数据滤波分析工具 - {os.path.basename(filename)}')
                
                # 显示成功消息
                self.statusBar.showMessage(f"成功加载文件: {os.path.basename(filename)}")
                logger.info("文件加载完成")
        except Exception as e:
            logger.error(f"加载CSV文件错误: {e}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "错误", f"加载文件时出错:\n{str(e)}")
            self.statusBar.showMessage("文件加载失败")

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
                        self.current_x_range = (x_min, x_max)
                        logger.info(f"设置X轴范围: {x_min} 到 {x_max}")
                    else:
                        logger.warning("X轴最小值应小于最大值")
                        self.statusBar.showMessage("X轴最小值应小于最大值")
                except ValueError:
                    logger.warning("X轴范围必须是数值")
                    self.statusBar.showMessage("X轴范围必须是数值")
            
            # 获取Y轴范围
            if self.y_min.text() and self.y_max.text():
                try:
                    y_min = float(self.y_min.text())
                    y_max = float(self.y_max.text())
                    if y_min < y_max:
                        self.ax.set_ylim(y_min, y_max)
                        self.current_y_range = (y_min, y_max)
                        logger.info(f"设置Y轴范围: {y_min} 到 {y_max}")
                    else:
                        logger.warning("Y轴最小值应小于最大值")
                        self.statusBar.showMessage("Y轴最小值应小于最大值")
                except ValueError:
                    logger.warning("Y轴范围必须是数值")
                    self.statusBar.showMessage("Y轴范围必须是数值")
            
            # 禁用自动范围
            if self.auto_range_check.isChecked():
                self.auto_range_check.setChecked(False)
            
            # 刷新画布
            self.canvas.draw()
            self.statusBar.showMessage("已应用坐标范围设置")
            logger.info("坐标轴范围已应用")
        except Exception as e:
            logger.error(f"应用坐标轴范围错误: {e}")
            logger.error(traceback.format_exc())
            self.statusBar.showMessage("应用坐标范围失败")

    def reset_zoom(self):
        """重置图表缩放"""
        try:
            logger.info("重置图表缩放")
            
            # 清除坐标范围输入框
            self.x_min.clear()
            self.x_max.clear()
            self.y_min.clear()
            self.y_max.clear()
            
            # 重置坐标轴范围
            self.ax.relim()  # 重新计算数据范围
            self.ax.autoscale_view()  # 自动调整视图
            
            # 如果启用了自动优化，则优化显示范围
            if self.auto_range:
                self.optimize_display_range()
            else:
                # 清除保存的范围
                self.current_x_range = None
                self.current_y_range = None
            
            # 刷新画布
            self.canvas.draw()
            self.statusBar.showMessage("已重置图表缩放")
            logger.info("图表缩放已重置")
        except Exception as e:
            logger.error(f"重置缩放错误: {e}")
            logger.error(traceback.format_exc())
            self.statusBar.showMessage("重置缩放失败")

    def darken_color(self, color):
        """使颜色变暗"""
        try:
            # 如果是颜色名称，转换为十六进制
            if color.startswith('#'):
                # 已经是十六进制
                hex_color = color
            else:
                # 尝试将颜色名称转换为十六进制
                try:
                    from matplotlib.colors import to_hex
                    hex_color = to_hex(color)
                except:
                    # 如果转换失败，使用默认颜色
                    logger.warning(f"无法将颜色 {color} 转换为十六进制，使用默认颜色")
                    return '#1f77b4'  # 默认蓝色
            
            # 移除井号
            hex_color = hex_color.lstrip('#')
            
            # 将十六进制颜色转换为RGB
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            # 使颜色变暗（乘以0.7）
            r = max(0, int(r * 0.7))
            g = max(0, int(g * 0.7))
            b = max(0, int(b * 0.7))
            
            # 转换回十六进制
            dark_hex = f'#{r:02x}{g:02x}{b:02x}'
            logger.info(f"颜色变暗: {color} -> {dark_hex}")
            return dark_hex
        except Exception as e:
            logger.error(f"颜色变暗处理错误: {e}")
            logger.error(traceback.format_exc())
            return '#1f77b4'  # 出错时返回默认蓝色

    def update_plot(self):
        """更新图表显示"""
        if self.waveform_data.raw_data is None:
            logger.warning("原始数据为空，无法更新图表")
            return
            
        try:
            logger.info("清除图表并开始绘制")
            self.ax.clear()
            self.ax2.clear()
            
            # 获取选中的波形（从内部列表）
            selected_waveforms = self.get_selected_waveforms()
            
            if not selected_waveforms:
                logger.info("未选择波形，图表更新完成")
                self.canvas.draw()
                return
                
            logger.info(f"选中波形数量: {len(selected_waveforms)}")
            
            # 保存当前坐标范围以便稍后恢复
            current_xlim = self.ax.get_xlim() if hasattr(self.ax, 'get_xlim') else None
            current_ylim = self.ax.get_ylim() if hasattr(self.ax, 'get_ylim') else None
            
            # 标记是否有保存的范围或当前范围
            has_range = (self.current_x_range is not None or current_xlim != (0, 1))
            
            # 为每个选中的波形绘制曲线
            for i, column in enumerate(selected_waveforms):
                logger.info(f"绘制波形 {i+1}/{len(selected_waveforms)}: {column}")
                
                # 使用波形样式
                if column not in self.waveform_data.styles:
                    logger.info(f"为波形 {column} 创建默认样式")
                    self.waveform_data.styles[column] = WaveformStyle()
                
                style = self.waveform_data.styles[column]
                logger.info(f"波形样式: 颜色={style.color}, 线型={style.line_style}, 线宽={style.line_width}, 高亮={style.highlighted}")
                
                # 获取数据
                try:
                    raw_data = self.waveform_data.raw_data[column].to_numpy()
                    logger.info(f"数据长度: {len(raw_data)}")
                except KeyError:
                    logger.error(f"列名 {column} 不存在")
                    continue
                except Exception as e:
                    logger.error(f"获取数据时出错: {e}")
                    continue
                
                # 根据是否高亮设置线条粗细
                line_width = style.line_width * 1.5 if style.highlighted else style.line_width
                
                # 检查列名是否包含滤波器名称（是否为滤波后的数据）
                is_filtered_data = False
                display_name = column
                
                # 检查是否是滤波后的数据通道
                for filter_type in [self._get_filter_display_name(ft) for ft in self.filter_menu_actions.keys()]:
                    if f"_{filter_type}" in column:
                        is_filtered_data = True
                        orig_name = column.split(f"_{filter_type}")[0]
                        display_name = f"{orig_name} ({filter_type})"
                        break
                
                # 绘制数据，使用自定义样式并启用拾取功能
                logger.info(f"绘制数据曲线: {display_name}")
                line = self.ax.plot(raw_data, 
                           label=display_name,
                           color=style.color,
                           linestyle=style.line_style,
                           linewidth=line_width,
                           picker=5)  # 开启拾取，容差为5像素
            
            # 设置标题和标签
            logger.info("设置图表标题和标签")
            self.ax.set_title('波形显示', fontsize=14)
            self.ax.set_xlabel('采样点', fontsize=12)
            self.ax.set_ylabel('数值', fontsize=12)
            
            # 添加图例
            logger.info("添加图例")
            self.ax.legend(loc='upper right', fontsize=10, framealpha=0.7)
            
            # 应用坐标范围
            if not self.auto_range:
                # 优先使用保存的范围
                if self.current_x_range is not None:
                    logger.info(f"应用保存的X轴范围: {self.current_x_range}")
                    self.ax.set_xlim(self.current_x_range)
                elif current_xlim and current_xlim != (0, 1):
                    logger.info(f"应用当前X轴范围: {current_xlim}")
                    self.ax.set_xlim(current_xlim)
                
                if self.current_y_range is not None:
                    logger.info(f"应用保存的Y轴范围: {self.current_y_range}")
                    self.ax.set_ylim(self.current_y_range)
                elif current_ylim and current_ylim != (0, 1):
                    logger.info(f"应用当前Y轴范围: {current_ylim}")
                    self.ax.set_ylim(current_ylim)
            else:
                # 自动范围模式，优化显示
                logger.info("自动优化显示范围")
                if not has_range:  # 只有在没有已存在范围时才优化
                    self.optimize_display_range()
                else:
                    # 有已存在范围，应用保存的范围
                    if self.current_x_range is not None:
                        self.ax.set_xlim(self.current_x_range)
                    if self.current_y_range is not None:
                        self.ax.set_ylim(self.current_y_range)
            
            self.figure.tight_layout()
            self.canvas.draw()
            logger.info("图表更新完成")
        except Exception as e:
            logger.error(f"绘图错误: {e}")
            logger.error(traceback.format_exc())
            QMessageBox.warning(self, "警告", f"更新图表时出错:\n{str(e)}")

    def show_style_menu(self, pos):
        """显示样式列表右键菜单"""
        try:
            selected_items = self.style_list.selectedItems()
            if not selected_items:
                return
            column = selected_items[0].text()
            if column not in self.waveform_data.styles:
                self.waveform_data.styles[column] = WaveformStyle()
            menu = QMenu(self)
            # 设置颜色
            color_action = menu.addAction("设置颜色")
            color_action.triggered.connect(lambda: self.set_waveform_color_from_menu(column))
            # 设置线型
            line_style_menu = menu.addMenu("设置线型")
            solid_action = line_style_menu.addAction("实线")
            solid_action.triggered.connect(lambda: self.set_waveform_line_style_from_menu(column, '-'))
            dashed_action = line_style_menu.addAction("虚线")
            dashed_action.triggered.connect(lambda: self.set_waveform_line_style_from_menu(column, '--'))
            dotted_action = line_style_menu.addAction("点线")
            dotted_action.triggered.connect(lambda: self.set_waveform_line_style_from_menu(column, ':'))
            dash_dot_action = line_style_menu.addAction("点划线")
            dash_dot_action.triggered.connect(lambda: self.set_waveform_line_style_from_menu(column, '-.'))
            # 设置线宽
            line_width_menu = menu.addMenu("设置线宽")
            thin_action = line_width_menu.addAction("细")
            thin_action.triggered.connect(lambda: self.set_waveform_line_width_from_menu(column, 1.0))
            normal_action = line_width_menu.addAction("中")
            normal_action.triggered.connect(lambda: self.set_waveform_line_width_from_menu(column, 1.5))
            thick_action = line_width_menu.addAction("粗")
            thick_action.triggered.connect(lambda: self.set_waveform_line_width_from_menu(column, 2.0))
            very_thick_action = line_width_menu.addAction("极粗")
            very_thick_action.triggered.connect(lambda: self.set_waveform_line_width_from_menu(column, 3.0))
            # 高亮
            highlight_action = menu.addAction("高亮波形")
            highlight_action.setCheckable(True)
            highlight_action.setChecked(self.waveform_data.styles[column].highlighted)
            highlight_action.triggered.connect(lambda: self.toggle_highlight_from_menu(column))
            # 弹出菜单
            menu.exec_(self.style_list.viewport().mapToGlobal(pos))
        except Exception as e:
            logger.error(f"显示样式菜单错误: {e}")
            logger.error(traceback.format_exc())

    def solo_highlight_waveform(self, column):
        """单独高亮选中的波形，降低其他波形亮度"""
        try:
            logger.info(f"单独高亮波形: {column}")
            # 将所有波形设置为非高亮，透明度降低
            for wave_column in self.waveform_data.styles:
                if wave_column == column:
                    # 高亮选中的波形
                    self.waveform_data.styles[wave_column].highlighted = True
                    self.waveform_data.styles[wave_column].line_width = 2.0  # 加粗
                else:
                    # 降低其他波形的显示
                    self.waveform_data.styles[wave_column].highlighted = False
                    self.waveform_data.styles[wave_column].line_width = 1.0  # 变细
            
            # 更新样式列表
            self.update_style_list()
            # 更新图表
            self.update_plot()
            self.statusBar.showMessage(f"单独高亮显示波形: {column}")
        except Exception as e:
            logger.error(f"单独高亮波形错误: {e}")
            logger.error(traceback.format_exc())
    
    def reset_all_highlights(self):
        """重置所有波形高亮状态"""
        try:
            logger.info("重置所有波形高亮状态")
            # 将所有波形设置为非高亮，但保持正常宽度
            for column in self.waveform_data.styles:
                self.waveform_data.styles[column].highlighted = False
                self.waveform_data.styles[column].line_width = 1.5  # 恢复默认宽度
            
            # 更新样式列表
            self.update_style_list()
            # 更新图表
            self.update_plot()
            self.statusBar.showMessage("已重置所有波形高亮状态")
        except Exception as e:
            logger.error(f"重置高亮状态错误: {e}")
            logger.error(traceback.format_exc())

    def toggle_auto_range(self, state):
        """切换自动范围设置"""
        try:
            self.auto_range = (state == Qt.Checked)
            logger.info(f"自动范围设置: {self.auto_range}")
            
            # 如果切换到自动模式，则重新计算并设置范围
            if self.auto_range:
                self.reset_zoom()
            
            # 更新坐标输入框的启用状态
            self.x_min.setEnabled(not self.auto_range)
            self.x_max.setEnabled(not self.auto_range)
            self.y_min.setEnabled(not self.auto_range)
            self.y_max.setEnabled(not self.auto_range)
            self.apply_range_btn.setEnabled(not self.auto_range)
            
            self.statusBar.showMessage("已" + ("启用" if self.auto_range else "禁用") + "自动范围优化")
        except Exception as e:
            logger.error(f"切换自动范围错误: {e}")
            logger.error(traceback.format_exc())
    
    def optimize_display_range(self):
        """优化显示范围，使波形更加清晰"""
        try:
            if not self.waveform_data.raw_data is None:
                logger.info("优化显示范围")
                
                # 获取当前选中的波形数据
                selected_items = self.waveform_list.selectedItems()
                if not selected_items:
                    return
                
                # 收集所有选中波形的数据点
                all_data = []
                for item in selected_items:
                    column = item.text()
                    data = self.waveform_data.raw_data[column].to_numpy()
                    all_data.extend(data)
                
                if not all_data:
                    return
                
                # 计算适合的Y轴范围（去除异常值）
                data_array = np.array(all_data)
                q1, q3 = np.percentile(data_array, [5, 95])  # 使用5%和95%分位数，减少异常值影响
                iqr = q3 - q1
                y_min = q1 - 0.5 * iqr
                y_max = q3 + 0.5 * iqr
                
                # 设置X轴范围（使用数据点索引）
                x_min = 0
                x_max = len(self.waveform_data.raw_data)
                
                # 保存当前范围
                self.current_x_range = (x_min, x_max)
                self.current_y_range = (y_min, y_max)
                
                # 更新输入框
                self.x_min.setText(str(int(x_min)))
                self.x_max.setText(str(int(x_max)))
                self.y_min.setText(f"{y_min:.2f}")
                self.y_max.setText(f"{y_max:.2f}")
                
                # 应用到图表
                self.ax.set_xlim(x_min, x_max)
                self.ax.set_ylim(y_min, y_max)
                
                logger.info(f"优化后范围: X=({x_min},{x_max}), Y=({y_min:.2f},{y_max:.2f})")
                
                # 刷新画布
                self.canvas.draw()
        except Exception as e:
            logger.error(f"优化显示范围错误: {e}")
            logger.error(traceback.format_exc())

    def toggle_status_selection(self, column, checked):
        """通过菜单切换状态数据选择状态"""
        try:
            # 在内部列表中设置选中状态
            for i in range(self.status_list.count()):
                item = self.status_list.item(i)
                if item.text() == column:
                    item.setSelected(checked)
                    break
            
            # 更新图表
            self.update_plot()
            
            if checked:
                self.statusBar.showMessage(f"已选择状态数据: {column}")
            else:
                self.statusBar.showMessage(f"已取消选择状态数据: {column}")
        except Exception as e:
            logger.error(f"切换状态数据选择错误: {e}")
            logger.error(traceback.format_exc())

    def _create_filter_panel(self, parent_layout):
        """创建滤波器设置面板"""
        try:
            logger.info("创建滤波器设置面板")
            
            # 创建滤波器设置组
            filter_group = QGroupBox("可用滤波器")
            filter_layout = QVBoxLayout()
            
            # 将可用滤波器以复选框形式列出
            filter_types = [
                'moving_average', 'median_filter', 'exponential_moving_average',
                'butterworth_filter', 'savitzky_golay', 'threshold_filter',
                'rate_limit_filter', 'group_average_filter'
            ]
            
            # 初始化滤波器复选框字典
            self.filter_checkboxes = {}
            
            # 确保filter_param_widgets已经初始化
            if not hasattr(self, 'filter_param_widgets'):
                self.filter_param_widgets = {}
            
            for filter_type in filter_types:
                # 创建复选框，与菜单项保持同步
                checkbox = QCheckBox(self._get_filter_display_name(filter_type))
                checkbox.stateChanged.connect(lambda state, ft=filter_type: self.toggle_filter_params(ft, state == Qt.Checked))
                
                # 如果已经在菜单中选中，同步选中状态
                if filter_type in self.filter_menu_actions:
                    checkbox.setChecked(self.filter_menu_actions[filter_type].isChecked())
                
                self.filter_checkboxes[filter_type] = checkbox
                filter_layout.addWidget(checkbox)
                
                # 创建参数面板
                param_widget = self._create_filter_param_widget(filter_type)
                param_widget.hide()  # 初始隐藏
                self.filter_param_widgets[filter_type] = param_widget
                filter_layout.addWidget(param_widget)
            
            filter_group.setLayout(filter_layout)
            
            # 创建操作按钮
            button_layout = QHBoxLayout()
            
            # 应用滤波器按钮
            self.apply_filters_btn = QPushButton("应用选中的滤波器")
            self.apply_filters_btn.clicked.connect(self.apply_selected_filters)
            button_layout.addWidget(self.apply_filters_btn)
            
            # 清除所有滤波器按钮
            self.clear_filters_btn = QPushButton("清除所有滤波器")
            self.clear_filters_btn.clicked.connect(self.clear_all_filters)
            button_layout.addWidget(self.clear_filters_btn)
            
            # 创建滚动区域来包含滤波器设置
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            filter_container = QWidget()
            filter_container_layout = QVBoxLayout(filter_container)
            filter_container_layout.addWidget(filter_group)
            filter_container_layout.addLayout(button_layout)
            scroll_area.setWidget(filter_container)
            
            # 设置最大高度，避免占用太多空间
            scroll_area.setMaximumHeight(400)
            
            # 添加到父布局
            parent_layout.addWidget(scroll_area)
            
            logger.info("滤波器设置面板创建完成")
        except Exception as e:
            logger.error(f"创建滤波器设置面板错误: {e}")
            logger.error(traceback.format_exc())

    def _get_filter_display_name(self, filter_type):
        """获取滤波器类型对应的显示名称"""
        try:
            filter_display_names = {
                'moving_average': '移动平均',
                'median_filter': '中值滤波',
                'exponential_moving_average': '指数移动平均',
                'butterworth_filter': '巴特沃斯滤波',
                'savitzky_golay': 'SG滤波',
                'threshold_filter': '限幅滤波',
                'rate_limit_filter': '增幅限制',
                'group_average_filter': '分组平均',
                'custom': '自定义公式'
            }
            
            return filter_display_names.get(filter_type, filter_type)
        except Exception as e:
            logger.error(f"获取滤波器显示名称错误: {e}")
            logger.error(traceback.format_exc())
            return filter_type  # 出错时返回原始名称

    def get_selected_waveforms(self):
        """获取选中的波形列表"""
        try:
            selected_waveforms = []
            # 从内部列表中获取选中的波形
            for i in range(self.waveform_list.count()):
                item = self.waveform_list.item(i)
                if item.isSelected():
                    selected_waveforms.append(item.text())
                    
            logger.info(f"获取到选中波形: {selected_waveforms}")
            return selected_waveforms
        except Exception as e:
            logger.error(f"获取选中波形错误: {e}")
            logger.error(traceback.format_exc())
            return []  # 出错时返回空列表

    def toggle_waveform_selection(self, column, checked):
        """通过菜单切换波形数据选择状态"""
        try:
            logger.info(f"切换波形'{column}'选择状态: {'选中' if checked else '取消选中'}")
            
            # 在内部列表中设置选中状态
            for i in range(self.waveform_list.count()):
                item = self.waveform_list.item(i)
                if item.text() == column:
                    item.setSelected(checked)
                    break
            
            # 更新图表
            self.update_plot()
            
            # 更新状态栏消息
            if checked:
                self.statusBar.showMessage(f"已选择波形: {column}")
            else:
                self.statusBar.showMessage(f"已取消选择波形: {column}")
        except Exception as e:
            logger.error(f"切换波形选择状态错误: {e}")
            logger.error(traceback.format_exc())

    def _create_menus(self):
        """创建菜单栏和菜单项"""
        try:
            logger.info("创建菜单栏")
            # 创建菜单栏
            menubar = self.menuBar()
            
            # 文件菜单
            file_menu = menubar.addMenu("文件")
            
            # 加载CSV文件动作
            load_csv_action = QAction("加载CSV文件", self)
            load_csv_action.triggered.connect(self.load_csv)
            file_menu.addAction(load_csv_action)
            
            # 退出动作
            exit_action = QAction("退出", self)
            exit_action.triggered.connect(self.close)
            file_menu.addAction(exit_action)
            
            # 波形选择菜单
            self.waveform_selection_menu = menubar.addMenu("波形选择")
            
            # 坐标设置菜单
            coordinate_menu = menubar.addMenu("坐标设置")
            
            # 应用坐标范围动作
            apply_range_action = QAction("应用坐标范围", self)
            apply_range_action.triggered.connect(self.apply_axis_range)
            coordinate_menu.addAction(apply_range_action)
            
            # 重置缩放动作
            reset_zoom_action = QAction("重置缩放", self)
            reset_zoom_action.triggered.connect(self.reset_zoom)
            coordinate_menu.addAction(reset_zoom_action)
            
            # 自动优化显示范围动作
            optimize_action = QAction("优化显示范围", self)
            optimize_action.triggered.connect(self.optimize_display_range)
            coordinate_menu.addAction(optimize_action)
            
            # 滤波器菜单
            filter_menu = menubar.addMenu("滤波器")
            
            # 为每种滤波器创建子菜单项
            filter_types = [
                'moving_average', 'median_filter', 'exponential_moving_average',
                'butterworth_filter', 'savitzky_golay', 'threshold_filter',
                'rate_limit_filter', 'group_average_filter'
            ]
            
            # 初始化滤波器菜单操作字典
            self.filter_menu_actions = {}
            
            # 为每个滤波器创建菜单项
            for filter_type in filter_types:
                # 创建可勾选的菜单项
                action = QAction(self._get_filter_display_name(filter_type), self)
                action.setCheckable(True)
                action.triggered.connect(lambda checked, ft=filter_type: self.toggle_filter_from_menu(ft, checked))
                filter_menu.addAction(action)
                
                # 保存菜单操作的引用
                self.filter_menu_actions[filter_type] = action
            
            filter_menu.addSeparator()
            
            # 应用滤波器动作
            apply_filters_action = QAction("应用选中的滤波器", self)
            apply_filters_action.triggered.connect(self.apply_selected_filters)
            filter_menu.addAction(apply_filters_action)
            
            # 滤波器参数设置
            filter_params_action = QAction("滤波器参数设置", self)
            filter_params_action.triggered.connect(self.show_filter_settings_dialog)
            filter_menu.addAction(filter_params_action)
            
            # 帮助菜单
            help_menu = menubar.addMenu("帮助")
            
            # 关于动作
            about_action = QAction("关于", self)
            about_action.triggered.connect(self.show_about)
            help_menu.addAction(about_action)
            
            logger.info("菜单栏创建完成")
        except Exception as e:
            logger.error(f"创建菜单栏错误: {e}")
            logger.error(traceback.format_exc())

    def show_filter_settings_dialog(self):
        """显示滤波器设置对话框"""
        try:
            logger.info("显示滤波器设置对话框")
            
            # 创建对话框
            dialog = QDialog(self)
            dialog.setWindowTitle("滤波器设置")
            dialog.setMinimumWidth(500)
            dialog.setMinimumHeight(600)
            
            # 创建布局
            layout = QVBoxLayout(dialog)
            
            # 创建滤波器组
            filter_group = QGroupBox("可用滤波器")
            filter_layout = QVBoxLayout()
            
            # 滤波器列表
            filter_types = [
                'moving_average', 'median_filter', 'exponential_moving_average',
                'butterworth_filter', 'savitzky_golay', 'threshold_filter',
                'rate_limit_filter', 'group_average_filter'
            ]
            
            # 保存对话框中的复选框和参数控件
            dialog_filter_checkboxes = {}
            dialog_param_widgets = {}
            
            # 创建滚动区域
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_widget = QWidget()
            scroll_layout = QVBoxLayout(scroll_widget)
            
            # 为每个滤波器创建复选框和参数控件
            for filter_type in filter_types:
                # 创建复选框
                checkbox = QCheckBox(self._get_filter_display_name(filter_type))
                
                # 如果已经在主界面选中，则在对话框中也选中
                if filter_type in self.filter_menu_actions:
                    checkbox.setChecked(self.filter_menu_actions[filter_type].isChecked())
                
                dialog_filter_checkboxes[filter_type] = checkbox
                scroll_layout.addWidget(checkbox)
                
                # 创建参数控件（如果已经存在，使用已有的；否则创建新的）
                if filter_type in self.filter_param_widgets and self.filter_param_widgets[filter_type]:
                    param_widget = self.filter_param_widgets[filter_type][0]
                else:
                    param_widget = self._create_filter_param_widget(filter_type)
                    self.filter_param_widgets[filter_type] = [param_widget]
                
                # 设置初始可见性
                param_widget.setVisible(checkbox.isChecked())
                
                # 连接信号
                checkbox.stateChanged.connect(lambda state, w=param_widget: w.setVisible(state))
                
                scroll_layout.addWidget(param_widget)
                scroll_layout.addSpacing(10)  # 添加间距
                
                # 保存参数控件引用
                dialog_param_widgets[filter_type] = param_widget
            
            # 设置滚动区域
            scroll_area.setWidget(scroll_widget)
            scroll_layout.addStretch(1)
            
            filter_group.setLayout(filter_layout)
            filter_layout.addWidget(scroll_area)
            layout.addWidget(filter_group)
            
            # 创建按钮
            button_box = QHBoxLayout()
            ok_button = QPushButton("确定")
            cancel_button = QPushButton("取消")
            
            ok_button.clicked.connect(dialog.accept)
            cancel_button.clicked.connect(dialog.reject)
            
            button_box.addWidget(ok_button)
            button_box.addWidget(cancel_button)
            layout.addLayout(button_box)
            
            # 显示对话框
            if dialog.exec_() == QDialog.Accepted:
                logger.info("应用滤波器设置")
                
                # 更新主界面上的复选框选中状态和菜单项选中状态
                for filter_type, checkbox in dialog_filter_checkboxes.items():
                    # 更新复选框状态
                    if filter_type in self.filter_checkboxes:
                        self.filter_checkboxes[filter_type].setChecked(checkbox.isChecked())
                    
                    # 更新菜单项状态
                    if filter_type in self.filter_menu_actions:
                        self.filter_menu_actions[filter_type].setChecked(checkbox.isChecked())
                
                self.statusBar.showMessage("滤波器设置已更新")
            else:
                logger.info("取消滤波器设置")
                
            logger.info("滤波器设置对话框处理完成")
        except Exception as e:
            logger.error(f"显示滤波器设置对话框错误: {e}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "错误", f"显示滤波器设置对话框时出错:\n{str(e)}")

    def toggle_filter_from_menu(self, filter_type, checked):
        """通过菜单切换滤波器的勾选状态"""
        try:
            logger.info(f"通过菜单切换滤波器 {filter_type} 的选中状态: {'选中' if checked else '取消选中'}")
            
            # 如果滤波器复选框已初始化，同步更新复选框的状态
            if hasattr(self, 'filter_checkboxes') and filter_type in self.filter_checkboxes:
                self.filter_checkboxes[filter_type].setChecked(checked)
            
            # 切换参数面板显示
            self.toggle_filter_params(filter_type, checked)
            
            # 更新状态栏消息
            display_name = self._get_filter_display_name(filter_type)
            if checked:
                self.statusBar.showMessage(f"已选择 {display_name} 滤波器")
            else:
                self.statusBar.showMessage(f"已取消选择 {display_name} 滤波器")
        except Exception as e:
            logger.error(f"切换滤波器选择状态错误: {e}")
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