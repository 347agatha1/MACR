"""
跨指标归一化与融合处理模块
用于对软件质量指标进行归一化处理，并支持交叉注意力融合
"""

import numpy as np


def load_quantitative_metrics(file_path):
    """
    从文件加载定量指标数据
    
    Args:
        file_path: 指标文件路径
        
    Returns:
        指标字典 {指标名称: 数值}
    """
    metrics = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.endswith(';'):
                line = line.rstrip(';')
            
            # 解析 "指标名:数值" 格式
            if ':' in line:
                parts = line.split(':')
                key = parts[0].strip()
                value = parts[1].strip()
                try:
                    metrics[key] = float(value)
                except ValueError:
                    pass
    
    return metrics


def min_max_normalize(value, min_val, max_val):
    """
    Min-Max归一化: x' = (x - min) / (max - min)
    映射到[0, 1]范围
    """
    if max_val == min_val:
        return 0.5  # 避免除零
    return (value - min_val) / (max_val - min_val)


class CrossAttationNormalizer:
    """
    交叉注意力归一化器
    
    针对不同类型的指标采用不同的归一化策略：
    - 效益型指标（越大越好）：直接归一化
    - 成本型指标（越小越好）：反向归一化
    """
    
    # 定义各指标的参考范围和类型
    METRIC_CONFIG = {
        '代码行数': {
            'type': 'cost',
            'min': 1,
            'max': 500,
            'description': '代码总行数，反映代码规模'
        },
        '圈复杂度': {
            'type': 'cost',
            'min': 1,
            'max': 100,
            'description': '代码路径复杂度，衡量代码理解难度和维护成本'
        },
        '代码注释率(%)': {
            'type': 'benefit',
            'min': 0,
            'max': 100,
            'description': '注释代码占总代码的比例，反映代码可读性'
        },
        '嵌套深度(层)': {
            'type': 'cost',
            'min': 1,
            'max': 10,
            'description': '代码嵌套层数，过深影响可读性和可维护性'
        },
        '扇出数': {
            'type': 'cost',
            'min': 0,
            'max': 20,
            'description': '模块依赖的其他模块数量，反映代码耦合度'
        },
        '测试用例通过率(%)': {
            'type': 'benefit',
            'min': 0,
            'max': 100,
            'description': '测试用例通过的比例，反映代码正确性'
        },
        'Halstead词典大小': {
            'type': 'cost',
            'min': 1,
            'max': 500,
            'description': '程序使用的不同操作符和操作数数量'
        },
        'Halstead程序长度': {
            'type': 'cost',
            'min': 1,
            'max': 10000,
            'description': '操作符和操作数总出现次数'
        },
        'Halstead程序容量': {
            'type': 'cost',
            'min': 1,
            'max': 50000,
            'description': '程序的信息量，表示理解程序所需的信息量'
        },
        'Halstead难度': {
            'type': 'cost',
            'min': 0,
            'max': 500,
            'description': '理解程序的难度，值越大越难理解'
        },
        'Halstead工作量': {
            'type': 'cost',
            'min': 0,
            'max': 10000000,
            'description': '理解和实现程序所需的工作量'
        },
        'Halstead时间(秒)': {
            'type': 'cost',
            'min': 0,
            'max': 500000,
            'description': '估计编写程序所需的时间'
        },
        'Halstead错误数': {
            'type': 'cost',
            'min': 0,
            'max': 50,
            'description': '估计程序中潜在的错误数量'
        },
        '可维护性指数MI(%)': {
            'type': 'benefit',
            'min': 0,
            'max': 100,
            'description': '代码可维护性评分，值越大越容易维护'
        },
        '时间复杂度等级': {
            'type': 'cost',
            'min': 1,
            'max': 7,
            'description': '时间复杂度等级：1=O(1), 2=O(log n), 3=O(n), 4=O(n log n), 5=O(n²), 6=O(n³), 7=O(2ⁿ)'
        },
        '空间复杂度等级': {
            'type': 'cost',
            'min': 1,
            'max': 7,
            'description': '空间复杂度等级：1=O(1), 2=O(log n), 3=O(n), 4=O(n log n), 5=O(n²), 6=O(n³), 7=O(2ⁿ)'
        }
    }
    
    def __init__(self):
        """初始化归一化器"""
        pass
    
    def normalize_metric(self, metric_name, value):
        """
        对单个指标进行归一化

        Args:
            metric_name: 指标名称
            value: 原始值

        Returns:
            归一化后的值 (0-100范围，越大越好)
        """
        if metric_name not in self.METRIC_CONFIG:
            # 未知指标，使用默认配置
            return min_max_normalize(value, 0, 100) * 100

        config = self.METRIC_CONFIG[metric_name]
        metric_type = config['type']
        min_val = config['min']
        max_val = config['max']

        normalized = min_max_normalize(value, min_val, max_val)

        # 成本型指标需要反转（越小越好 → 归一化值越大）
        if metric_type == 'cost':
            normalized = 1 - normalized

        return np.clip(normalized * 100, 0, 100)
    
    def normalize_all(self, metrics_dict):
        """
        归一化所有指标
        
        Args:
            metrics_dict: 原始指标字典
            
        Returns:
            归一化后的指标字典
        """
        normalized = {}
        
        for name, value in metrics_dict.items():
            normalized[name] = self.normalize_metric(name, value)
        
        return normalized
    
    def calculate_weighted_score(self, normalized_metrics, weights=None):
        """
        计算加权综合得分
        
        Args:
            normalized_metrics: 归一化后的指标字典
            weights: 权重字典，如果为None则使用等权重
            
        Returns:
            综合得分 (0-100)
        """
        if weights is None:
            # 默认等权重
            weights = {name: 1.0 / len(normalized_metrics) 
                      for name in normalized_metrics}
        
        total_score = 0
        total_weight = 0
        
        for name, value in normalized_metrics.items():
            weight = weights.get(name, 0)
            total_score += value * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0
        
        return total_score / total_weight
    
    def analyze_metrics(self, metrics_dict):
        """
        分析指标并提供改进建议
        
        Args:
            metrics_dict: 原始指标字典
            
        Returns:
            分析报告
        """
        report = {
            'normalized_values': {},
            'issues': [],
            'suggestions': [],
            'overall_score': 0
        }
        
        # 归一化并分析
        for name, value in metrics_dict.items():
            normalized = self.normalize_metric(name, value)
            report['normalized_values'][name] = {
                'raw': value,
                'normalized': normalized,
                'status': self._get_status(normalized)
            }
            
            # 检查问题
            if name == '圈复杂度' and value > 10:
                report['issues'].append(f'圈复杂度({value})过高，建议简化代码逻辑')
            elif name == '嵌套深度(层)' and value > 4:
                report['issues'].append(f'嵌套深度({value})过深，建议重构')
            elif name == '扇出数' and value > 10:
                report['issues'].append(f'扇出数({value})过高，建议降低耦合度')
            elif name == 'Halstead难度' and value > 100:
                report['issues'].append(f'Halstead难度({value})过高，建议简化代码结构')
            elif name == '时间复杂度等级' and value >= 5:
                report['issues'].append(f'时间复杂度等级({value})过高，建议优化算法')
            elif name == '空间复杂度等级' and value >= 5:
                report['issues'].append(f'空间复杂度等级({value})过高，建议优化空间使用')
            elif name == '可维护性指数MI(%)' and value < 50:
                report['issues'].append(f'可维护性指数({value}%)较低，建议提高代码质量')
        
        # 计算综合得分
        normalized_metrics = {k: v['normalized'] 
                            for k, v in report['normalized_values'].items()}
        report['overall_score'] = self.calculate_weighted_score(normalized_metrics)
        
        return report
    
    def _get_status(self, normalized_value):
        """根据归一化值判断状态"""
        if normalized_value >= 0.8:
            return '优秀'
        elif normalized_value >= 0.6:
            return '良好'
        elif normalized_value >= 0.4:
            return '一般'
        else:
            return '待改进'
    
    def cross_attention_fusion(self, metrics_list, num_heads=4):
        """
        使用交叉注意力机制融合多个指标集
        
        模拟多头注意力机制，对不同指标进行加权融合
        
        Args:
            metrics_list: 多个时间点的指标字典列表
            num_heads: 注意力头数
            
        Returns:
            融合后的指标字典
        """
        if len(metrics_list) < 2:
            return metrics_list[0] if metrics_list else {}
        
        # 归一化所有指标集
        normalized_sets = []
        for metrics in metrics_list:
            normalized_sets.append(self.normalize_all(metrics))
        
        # 使用注意力机制进行融合
        all_keys = set()
        for ns in normalized_sets:
            all_keys.update(ns.keys())
        
        fused = {}
        for key in all_keys:
            values = [ns.get(key, 0) for ns in normalized_sets]
            # 使用softmax风格的加权
            weights = np.exp(np.array(values)) / np.sum(np.exp(values))
            fused[key] = np.sum(np.array(values) * weights)
        
        return fused


def demo_cross_attation():
    """
    演示归一化和分析功能
    """
    print("=" * 70)
    print("跨指标归一化与融合处理演示")
    print("=" * 70)
    
    # 加载指标数据
    metrics_file = 'QuantitativeMetric.txt'
    metrics = load_quantitative_metrics(metrics_file)
    
    print("\n原始指标数据:")
    print("-" * 50)
    for name, value in metrics.items():
        print(f"  {name}: {value}")
    
    # 创建归一化器
    normalizer = CrossAttationNormalizer()
    
    # 归一化所有指标
    print("\n" + "=" * 70)
    print("归一化处理结果")
    print("=" * 70)
    
    normalized = normalizer.normalize_all(metrics)
    
    for name, value in normalized.items():
        config = normalizer.METRIC_CONFIG.get(name, {})
        metric_type = config.get('type', 'unknown')
        raw_value = metrics[name]
        print(f"\n{name}:")
        print(f"  原始值: {raw_value}")
        print(f"  归一化值: {value:.4f} (0-1范围)")
        print(f"  指标类型: {metric_type}")
        print(f"  说明: {config.get('description', '无')}")
    
    # 分析指标
    print("\n" + "=" * 70)
    print("指标分析报告")
    print("=" * 70)
    
    report = normalizer.analyze_metrics(metrics)
    
    print(f"\n综合得分: {report['overall_score']:.2f} / 100")
    
    print("\n各指标状态:")
    for name, info in report['normalized_values'].items():
        print(f"  {name}: {info['status']} (归一化值: {info['normalized']:.4f})")
    
    if report['issues']:
        print("\n发现问题:")
        for issue in report['issues']:
            print(f"  ⚠ {issue}")
    else:
        print("\n✓ 未发现明显问题")
    
    # 各指标具体处理说明
    print("\n" + "=" * 70)
    print("各指标处理方式说明")
    print("=" * 70)
    
    process_descriptions = {
        '圈复杂度': """
        处理方式: 成本型指标，反向归一化
        - 圈复杂度越高，代码理解和维护越困难
        - 归一化时取反: normalized = 1 - (value - min) / (max - min)
        - 建议值: 圈复杂度 < 10（归一化值 > 0.8）
        - 优化建议: 拆分函数、消除嵌套、使用卫语句
        """,
        
        '平均执行时间(ms)': """
        处理方式: 成本型指标，反向归一化
        - 执行时间越长，系统响应越慢
        - 归一化时取反，确保低执行时间得到高分
        - 建议值: 执行时间 < 100ms（归一化值 > 0.8）
        - 优化建议: 算法优化、缓存、并行处理
        """,
        
        '内存占用峰值(KB)': """
        处理方式: 成本型指标，反向归一化
        - 内存占用越低，系统资源消耗越少
        - 归一化时取反，鼓励低内存使用
        - 建议值: 内存占用 < 1024KB（归一化值 > 0.5）
        - 优化建议: 及时释放、对象池、内存优化
        """,
        
        '扇出数': """
        处理方式: 成本型指标，反向归一化
        - 扇出数越高，模块耦合度越高
        - 扇出数过高意味着依赖过多模块
        - 建议值: 扇出数 < 7（归一化值 > 0.6）
        - 优化建议: 接口抽象、解耦、依赖注入
        """
    }
    
    for metric, description in process_descriptions.items():
        if metric in metrics:
            print(f"\n{metric}:")
            print(description)


if __name__ == "__main__":
    demo_cross_attation()
