import os
import ast
import sys
import re
import io
import math
import token
import tokenize
import keyword
import tempfile
import threading
from typing import Dict, List, Tuple, Any
import openpyxl

try:
    from radon.complexity import cc_visit
    from radon.metrics import h_visit, mi_visit
except ImportError:
    cc_visit = None
    h_visit = None
    mi_visit = None


class TimeoutException(Exception):
    """超时异常"""
    pass


def run_with_timeout(func, timeout_seconds):
    """
    在指定时间内运行函数，超时则抛出TimeoutException
    
    Args:
        func: 要执行的函数
        timeout_seconds: 超时时间（秒）
        
    Returns:
        函数执行结果
    """
    result = [None]
    exception = [None]
    
    def worker():
        try:
            result[0] = func()
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        raise TimeoutException(f"执行超时（{timeout_seconds}秒）")
    
    if exception[0]:
        raise exception[0]
    
    return result[0]


class MLCodeEvaluator:
    """机器学习代码定量指标评价器"""
    
    def __init__(self):
        """初始化评价器"""
        # 指标目标阈值（基础代码质量指标 + Halstead 指标 + 复杂度指标）
        self.target_thresholds = {
            'cyclomatic_complexity': {'value': 8.0, 'condition': '<='},
            'comment_rate': {'value': 50.0, 'condition': '>='},
            'nesting_depth': {'value': 2.0, 'condition': '<='},
            'fan_out': {'value': 5.0, 'condition': '<='},
            'test_pass_rate': {'value': 80.0, 'condition': '>='},
            'code_lines': {'value': 50.0, 'condition': '<='},
            # Halstead 指标阈值
            'halstead_vocabulary': {'value': 50.0, 'condition': '<='},   # 程序词典大小
            'halstead_length': {'value': 200.0, 'condition': '<='},      # 程序实际长度
            'halstead_volume': {'value': 1000.0, 'condition': '<='},     # 程序容量
            'halstead_difficulty': {'value': 15.0, 'condition': '<='},   # 程序难度
            'halstead_effort': {'value': 10000.0, 'condition': '<='},    # 工作量
            'halstead_time': {'value': 600.0, 'condition': '<='},        # 时间（秒）
            'halstead_bugs': {'value': 1.0, 'condition': '<='},          # 估计错误数
            'maintainability_index': {'value': 65.0, 'condition': '>='}, # 可维护性指数
            # 时间/空间复杂度等级（数值越小越好，对应 O(1)=1, O(log n)=2, O(n)=3, O(n log n)=4, O(n²)=5, O(n³)=6, O(2ⁿ)=7）
            'time_complexity': {'value': 4.0, 'condition': '<='},        # 时间复杂度等级
            'space_complexity': {'value': 3.0, 'condition': '<='}        # 空间复杂度等级
        }
        
        # 核心功能映射（用于识别代码类型）
        self.code_type_patterns = {
            '线性回归': ['linear', 'LinearRegression', 'regression', 'least squares'],
            '逻辑回归': ['logistic', 'LogisticRegression', 'sigmoid', 'classifier'],
            'KNN分类器': ['knn', 'KNN', 'KNeighbors', 'nearest neighbor'],
            '决策树': ['decision', 'DecisionTree', 'tree', 'ID3', 'C4.5'],
            '随机森林': ['random', 'forest', 'RandomForest', 'ensemble'],
            'SVM': ['svm', 'SVM', 'SupportVector', 'kernel'],
            '聚类算法': ['cluster', 'kmeans', 'KMeans', 'DBSCAN', 'hierarchical'],
            '神经网络': ['neural', 'network', 'nn', 'MLP', 'perceptron'],
            'PCA': ['pca', 'PCA', 'Principal', 'component'],
            '其他机器学习算法': []
        }
    
    def _get_tokens(self, code: str) -> List[tokenize.TokenInfo]:
        """使用 tokenize 获取源码 token，减少文本匹配误差。"""
        return list(tokenize.generate_tokens(io.StringIO(code).readline))

    def _get_ast_tree(self, code: str) -> ast.AST:
        """统一 AST 解析入口。"""
        return ast.parse(code)

    def _get_logical_loc(self, code: str) -> int:
        """基于 token 统计逻辑代码行数。"""
        code_lines = set()
        for tok in self._get_tokens(code):
            if tok.type in {
                tokenize.ENCODING,
                tokenize.ENDMARKER,
                tokenize.NEWLINE,
                tokenize.NL,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.COMMENT,
            }:
                continue
            code_lines.add(tok.start[0])
        return len(code_lines)

    def _get_comment_lines(self, code: str) -> int:
        """基于 token 统计注释行数。"""
        comment_lines = set()
        for tok in self._get_tokens(code):
            if tok.type == tokenize.COMMENT:
                comment_lines.add(tok.start[0])
        return len(comment_lines)

    def _get_docstring_lines(self, code: str) -> int:
        """统计模块、类、函数文档字符串占用的行数。"""
        tree = self._get_ast_tree(code)
        doc_lines = set()

        def mark_docstring(node: ast.AST) -> None:
            body = getattr(node, 'body', None)
            if not body:
                return
            first_stmt = body[0]
            if not isinstance(first_stmt, ast.Expr):
                return
            value = first_stmt.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                start = getattr(first_stmt, 'lineno', None)
                end = getattr(first_stmt, 'end_lineno', start)
                if start is not None and end is not None:
                    doc_lines.update(range(start, end + 1))

        mark_docstring(tree)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                mark_docstring(node)

        return len(doc_lines)

    def _get_operator_and_operand_tokens(self, code: str) -> Tuple[List[str], List[str]]:
        """基于 tokenize 提取 Halstead 所需的操作符和操作数。"""
        operators = []
        operands = []
        control_keywords = {
            'if', 'elif', 'else', 'for', 'while', 'break', 'continue',
            'return', 'yield', 'try', 'except', 'finally', 'raise',
            'with', 'async', 'await', 'lambda', 'import', 'from',
            'pass', 'del', 'assert', 'and', 'or', 'not', 'in', 'is',
            'match', 'case'
        }

        for tok in self._get_tokens(code):
            if tok.type in {
                tokenize.ENCODING,
                tokenize.ENDMARKER,
                tokenize.NEWLINE,
                tokenize.NL,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.COMMENT,
            }:
                continue

            if tok.type == token.OP:
                operators.append(tok.string)
            elif tok.type == token.NAME:
                if tok.string in control_keywords or keyword.iskeyword(tok.string):
                    operators.append(tok.string)
                else:
                    operands.append(tok.string)
            elif tok.type in {token.NUMBER, token.STRING}:
                operands.append(tok.string)

        return operators, operands

    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """
        校验文件合法性
        
        Args:
            file_path: 文件路径
            
        Returns:
            (是否合法, 错误信息/成功信息)
        """
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return False, "文件不存在"
        
        # 检查是否是Python文件
        if not file_path.endswith('.py'):
            return False, "不是Python文件"
        
        # 检查文件是否可读
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if not content.strip():
                return False, "文件内容为空"
        except Exception as e:
            return False, f"文件无法读取: {str(e)}"
        
        # 检查是否是有效的Python代码
        try:
            ast.parse(content)
        except SyntaxError as e:
            return False, f"Python语法错误: {str(e)}"
        
        return True, "文件校验通过"
    
    def parse_code(self, file_path: str) -> Dict[str, Any]:
        """
        解析代码，提取关键信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            解析结果字典
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # 代码文本
        lines = code.split('\n')
        
        # 提取导入的库
        imports = []
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        # 提取类和函数注意
        classes = []
        functions = []
        core_functions = []  # 核心函数（fit, predict等）
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append({
                    'name': node.name,
                    'lineno': node.lineno,
                    'end_lineno': getattr(node, 'end_lineno', node.lineno)
                })
            elif isinstance(node, ast.FunctionDef):
                func_info = {
                    'name': node.name,
                    'lineno': node.lineno,
                    'end_lineno': getattr(node, 'end_lineno', node.lineno)
                }
                functions.append(func_info)
                
                # 检查是否是核心函数
                core_func_names = ['fit', 'predict', 'train', 'evaluate', 'test', 'score']
                if node.name.lower() in core_func_names:
                    core_functions.append(func_info)
        
        # 判断代码类型
        code_type = '其他机器学习算法'
        code_lower = code.lower()
        for algo_type, patterns in self.code_type_patterns.items():
            if algo_type != '其他机器学习算法' and any(pattern.lower() in code_lower for pattern in patterns):
                code_type = algo_type
                break
        
        # 生成功能概述
        core_func_names = [f['name'] for f in core_functions]
        if 'fit' in core_func_names and 'predict' in core_func_names:
            function_overview = f"实现了{code_type}的核心功能，包括模型训练(fit)和预测(predict)"
        elif 'fit' in core_func_names:
            function_overview = f"实现了{code_type}的模型训练功能(fit)"
        elif 'predict' in core_func_names:
            function_overview = f"实现了{code_type}的预测功能(predict)"
        else:
            function_overview = f"实现了{code_type}的基础功能"
        
        return {
            'file_path': file_path,
            'code': code,
            'lines': lines,
            'imports': imports,
            'classes': classes,
            'functions': functions,
            'core_functions': core_functions,
            'code_type': code_type,
            'function_overview': function_overview
        }
    
    def calculate_metrics(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算各项评价指标（仅使用真实计算的指标，无模拟/估算）
        
        Args:
            parsed_data: 解析后的数据
            
        Returns:
            指标计算结果
        """
        metrics = {}
        
        # 1. 代码行数
        code_lines = self._calculate_code_lines(parsed_data['code'])
        metrics['code_lines'] = code_lines
        
        # 2. 注释率计算
        comment_rate = self._calculate_comment_rate(parsed_data['code'])
        metrics['comment_rate'] = comment_rate
        
        # 3. 圈复杂度计算
        cyclomatic_complexity = self._legacy_calculate_cyclomatic_complexity(parsed_data['code'])
        metrics['cyclomatic_complexity'] = cyclomatic_complexity
        
        # 4. 嵌套深度计算
        nesting_depth = self._legacy_calculate_nesting_depth(parsed_data['code'])
        metrics['nesting_depth'] = nesting_depth
        
        # 5. 扇出数计算
        fan_out = self._legacy_calculate_fan_out(parsed_data['code'])
        metrics['fan_out'] = fan_out
        
        # 6. Halstead 指标计算（全部真实计算）
        halstead = self._legacy_calculate_halstead_metrics(parsed_data['code'])
        metrics.update(halstead)
        
        # 7. 可维护性指数 MI（基于 Halstead 体积、圈复杂度、代码行数）
        mi = self._legacy_calculate_maintainability_index(
            halstead['halstead_volume'],
            metrics['cyclomatic_complexity'],
            metrics['code_lines']
        )
        metrics['maintainability_index'] = mi
        
        # 8. 时间复杂度和空间复杂度分析
        complexity = self._legacy_calculate_complexity_metrics(parsed_data['code'])
        metrics.update(complexity)
        
        # 评估达标情况
        passed_results = {}
        for metric_name, value in metrics.items():
            if metric_name in self.target_thresholds:
                threshold = self.target_thresholds[metric_name]
                if threshold['condition'] == '>=':
                    passed_results[f'{metric_name}_passed'] = value >= threshold['value']
                elif threshold['condition'] == '<=':
                    passed_results[f'{metric_name}_passed'] = value <= threshold['value']
                elif threshold['condition'] == '==':
                    passed_results[f'{metric_name}_passed'] = abs(value - threshold['value']) < 1e-6
        
        # 将结果合并到原字典
        metrics.update(passed_results)
        
        # 生成优化建议
        metrics['optimization_suggestions'] = self._generate_optimization_suggestions(metrics, parsed_data)
        
        # 生成综合评价
        metrics['overall_rating'] = self._calculate_overall_rating(metrics)
        
        return metrics
    
    def _calculate_halstead_metrics(self, code: str) -> Dict[str, float]:
        """
        计算完整的 Halstead 指标（全部真实计算）
        
        Halstead 指标基于代码中的操作符(operators)和操作数(operands)数量：
        - n₁: 不同操作符数量
        - n₂: 不同操作数数量  
        - N₁: 操作符总出现次数
        - N₂: 操作数总出现次数
        
        衍生公式：
        - 程序词典大小: n = n₁ + n₂
        - 程序实际长度: N = N₁ + N₂
        - 程序容量: V = N × log₂(n)
        - 程序难度: D = (n₁/2) × (N₂/n₂)
        - 工作量: E = D × V
        - 时间: T = E / 18
        - 错误数: B = V / 3000
        """
        # 定义 Python 操作符模式（基于 Halstead 理论分类）
        operator_patterns = [
            # 算术操作符
            r'\+', r'-', r'\*', r'/', r'//', r'\%', r'\*\*',
            # 比较操作符
            r'==', r'!=', r'<', r'>', r'<=', r'>=',
            # 逻辑操作符
            r'and', r'or', r'not',
            # 赋值操作符
            r'=', r'\+=', r'-=', r'\*=', r'/=', r'//=', r'\%=', r'\*\*=',
            # 位操作符
            r'\|', r'&', r'^', r'~', r'<<', r'>>',
            # 身份操作符
            r'is', r'is not',
            # 成员操作符
            r'in', r'not in',
            # 控制流关键字
            r'\bif\b', r'\belif\b', r'\belse\b',
            r'\bfor\b', r'\bwhile\b', r'\bbreak\b', r'\bcontinue\b',
            r'\breturn\b', r'\byield\b',
            # 异常处理
            r'\btry\b', r'\bexcept\b', r'\bfinally\b', r'\braise\b',
            # 类/函数定义
            r'\bdef\b', r'\bclass\b', r'\blambda\b',
            # 导入
            r'\bimport\b', r'\bfrom\b',
            # 其他关键字
            r'\bpass\b', r'\bdel\b', r'\bassert\b',
            # 括号和分隔符
            r'\(', r'\)', r'\[', r'\]', r'\{', r'\}',
            r',', r':', r'.', r'->', r'@'
        ]
        
        # 提取操作符
        operators = []
        for pattern in operator_patterns:
            matches = re.findall(pattern, code)
            operators.extend(matches)
        
        # 提取操作数（变量名、常量、函数调用等）
        operand_patterns = [
            r'\b[a-zA-Z_][a-zA-Z0-9_]*\b',  # 标识符
            r'\b\d+\.?\d*\b',                # 数字常量
            r'"[^"]*"',                      # 字符串常量
            r"'[^']*'",                      # 字符串常量
            r'""".*?"""',                    # 多行字符串
            r"'''.*?'''",                    # 多行字符串
            r'\bTrue\b', r'\bFalse\b', r'\bNone\b'  # 特殊常量
        ]
        
        operands = []
        for pattern in operand_patterns:
            matches = re.findall(pattern, code, re.DOTALL)
            operands.extend(matches)
        
        # 计算核心指标
        n1 = len(set(operators))  # 不同操作符数量
        n2 = len(set(operands))   # 不同操作数数量
        N1 = len(operators)       # 操作符总出现次数
        N2 = len(operands)        # 操作数总出现次数
        
        # 计算衍生指标
        n = n1 + n2                              # 程序词典大小
        N = N1 + N2                              # 程序实际长度
        
        # 程序容量 V = N × log₂(n)
        if n > 1:
            V = N * (n.bit_length() - 1)  # log2(n) ≈ bit_length - 1
            # 更精确的计算
            import math
            V = N * math.log2(n)
        else:
            V = 0.0
        
        # 程序难度 D = (n₁/2) × (N₂/n₂) 【注意：是 N₂/n₂，不是 n₂/N₂】
        if n2 > 0 and N2 > 0:
            D = (n1 / 2) * (N2 / n2)
        else:
            D = 0.0
        
        # 工作量 E = D × V
        E = D * V
        
        # 时间 T = E / 18（秒）
        T = E / 18
        
        # 错误数 B = V / 3000
        B = V / 3000
        
        return {
            'halstead_n1': round(n1, 2),          # 不同操作符数量
            'halstead_n2': round(n2, 2),          # 不同操作数数量
            'halstead_N1': round(N1, 2),          # 操作符总出现次数
            'halstead_N2': round(N2, 2),          # 操作数总出现次数
            'halstead_vocabulary': round(n, 2),   # 程序词典大小
            'halstead_length': round(N, 2),       # 程序实际长度
            'halstead_volume': round(V, 2),       # 程序容量
            'halstead_difficulty': round(D, 2),   # 程序难度
            'halstead_effort': round(E, 2),       # 工作量
            'halstead_time': round(T, 2),         # 时间（秒）
            'halstead_bugs': round(B, 4)          # 估计错误数
        }
    
    def _calculate_maintainability_index(self, volume: float, cyclomatic_complexity: float, code_lines: int) -> float:
        """
        计算可维护性指数 MI（Maintainability Index）
        
        使用 Microsoft Visual Studio 公式：
        MI = max(0, (171 - 5.2 * ln(volume) - 0.23 * cyclomatic_complexity - 16.2 * ln(code_lines)) * 100 / 171)
        
        其中：
        - volume: Halstead 程序容量
        - cyclomatic_complexity: 圈复杂度
        - code_lines: 代码行数
        
        结果范围：0-100，数值越高表示代码越容易维护
        """
        import math
        
        if volume <= 0 or code_lines <= 0:
            return 0.0
        
        # 计算各部分
        ln_volume = math.log(volume)
        ln_lines = math.log(code_lines)
        
        # Microsoft 公式
        mi = 171 - 5.2 * ln_volume - 0.23 * cyclomatic_complexity - 16.2 * ln_lines
        
        # 归一化到 0-100 范围
        mi = mi * 100 / 171
        
        # 确保结果在 0-100 之间
        mi = max(0.0, min(100.0, mi))
        
        return round(mi, 2)
    
    def _calculate_complexity_metrics(self, code: str) -> Dict[str, float]:
        """
        计算时间复杂度和空间复杂度
        
        复杂度等级映射：
        - 1: O(1) - 常数时间
        - 2: O(log n) - 对数时间
        - 3: O(n) - 线性时间
        - 4: O(n log n) - 线性对数时间
        - 5: O(n²) - 平方时间
        - 6: O(n³) - 立方时间
        - 7: O(2ⁿ) - 指数时间
        
        时间复杂度分析：
        - 分析循环嵌套深度
        - 识别递归调用
        - 识别哈希表操作（dict）优化
        - 识别常见算法模式（排序、搜索等）
        
        空间复杂度分析：
        - 分析数据结构创建
        - 分析递归调用深度
        - 分析动态内存分配
        """
        # 时间复杂度分析
        time_complexity = 1  # 默认 O(1)
        
        # 分析循环结构（使用AST精确统计）
        tree = ast.parse(code)
        loop_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.For) or isinstance(node, ast.While):
                loop_count += 1
        
        # 分析嵌套深度（基于缩进）
        lines = code.split('\n')
        max_nesting = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            
            indent = len(line) - len(stripped)
            level = indent // 4
            
            if any(keyword in stripped for keyword in ['for ', 'while ']):
                max_nesting = max(max_nesting, level)
        
        # 分析递归调用
        has_recursion = False
        functions = re.findall(r'\bdef\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code)
        
        for func_name in functions:
            func_pattern = rf'def\s+{func_name}\s*\([^)]*\):'
            match = re.search(func_pattern, code)
            if match:
                start_pos = match.end()
                next_func = re.search(r'\bdef\s+\w+\s*\(', code[start_pos:])
                if next_func:
                    end_pos = start_pos + next_func.start()
                else:
                    end_pos = len(code)
                
                func_body = code[start_pos:end_pos]
                if func_name in func_body:
                    has_recursion = True
                    break
        
        # 识别常见算法模式
        algorithm_patterns = {
            'sorting': [r'\bsorted\s*\(', r'\.sort\s*\(', r'\bquickSort\b', r'\bmergeSort\b', r'\bheapSort\b'],
            'binary_search': [r'\bbinary_search\b', r'\bbisect\s*\('],
            'matrix_operation': [r'\bmatrix\b', r'\[\[.*\]\]', r'\bshape\b'],
            'exponential': [r'\*\*.*n', r'\bpower\s*\(', r'\bfactorial\b', r'\bfibonacci\b']
        }
        
        has_matrix = any(re.search(pattern, code) for pattern in algorithm_patterns['matrix_operation'])
        has_exponential = any(re.search(pattern, code) for pattern in algorithm_patterns['exponential'])
        has_sorting = any(re.search(pattern, code) for pattern in algorithm_patterns['sorting'])
        has_binary_search = any(re.search(pattern, code) for pattern in algorithm_patterns['binary_search'])
        
        # 识别哈希表操作（用于优化复杂度判断）
        has_dict_usage = any(pattern in code for pattern in ['{}', 'dict()', 'seen[', 'map[', 'hash[', '.get('])
        
        # 判断是否有真正的嵌套循环（排除哈希表优化的情况）
        has_nested_loop = False
        if max_nesting >= 2:
            # 检查内层是否是真正的循环，而不是if语句
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith('for ') or stripped.startswith('while '):
                    indent = len(line) - len(stripped)
                    level = indent // 4
                    if level >= 2:
                        has_nested_loop = True
                        break
        
        # 综合评估时间复杂度
        if has_exponential or (has_recursion and 'fibonacci' in code.lower()):
            time_complexity = 7  # O(2ⁿ)
        elif has_matrix or (max_nesting >= 3):
            time_complexity = 6  # O(n³)
        elif has_nested_loop and not has_dict_usage:
            time_complexity = 5  # O(n²)
        elif has_sorting or ((loop_count >= 2) and not has_binary_search):
            time_complexity = 4  # O(n log n)
        elif loop_count >= 1 or has_recursion:
            time_complexity = 3  # O(n)
        elif has_binary_search:
            time_complexity = 2  # O(log n)
        else:
            time_complexity = 1  # O(1)
        
        # 空间复杂度分析
        space_complexity = 1  # 默认 O(1)
        
        # 分析数据结构创建
        data_structure_patterns = [
            r'\[\s*\]',                # 列表创建
            r'\{\s*\}',                # 字典创建
            r'\bList\s*\[',            # 类型提示列表
            r'\bDict\s*\[',            # 类型提示字典
            r'np\.\w+\s*\(',           # numpy 数组创建
            r'pd\.\w+\s*\(',           # pandas 数据结构
            r'\bset\s*\(',             # 集合创建
            r'\btuple\s*\('            # 元组创建
        ]
        
        data_structure_count = sum(len(re.findall(pattern, code)) for pattern in data_structure_patterns)
        
        # 分析递归深度（影响调用栈空间）
        if has_recursion:
            # 估计递归深度
            # 如果有递归且有循环，可能是更深的递归
            if loop_count >= 1:
                space_complexity = max(space_complexity, 4)  # O(n) 或更高
            else:
                space_complexity = max(space_complexity, 3)  # O(n)
        
        # 分析动态内存分配
        has_dynamic_allocation = any([
            r'\bappend\s*\(' in code,
            r'\bextend\s*\(' in code,
            r'\binsert\s*\(' in code,
            r'\bpop\s*\(' in code,
            r'\bremove\s*\(' in code,
            r'\bdel\s+' in code
        ])
        
        # 综合评估空间复杂度
        if data_structure_count >= 5 or (has_matrix and data_structure_count >= 2):
            space_complexity = 5  # O(n²) - 二维数据结构
        elif data_structure_count >= 3 or has_recursion:
            space_complexity = 3  # O(n) - 线性空间
        elif data_structure_count >= 1 or has_dynamic_allocation:
            space_complexity = 2  # O(log n) 或 O(1) 的变体
        else:
            space_complexity = 1  # O(1) - 常数空间
        
        # 转换为可读的复杂度表示
        complexity_labels = {
            1: 'O(1)',
            2: 'O(log n)',
            3: 'O(n)',
            4: 'O(n log n)',
            5: 'O(n²)',
            6: 'O(n³)',
            7: 'O(2^n)'
        }
        
        return {
            'time_complexity': float(time_complexity),
            'time_complexity_label': complexity_labels[time_complexity],
            'space_complexity': float(space_complexity),
            'space_complexity_label': complexity_labels[space_complexity]
        }
    
    def _legacy_calculate_code_lines(self, lines: List[str]) -> int:
        """计算有效代码行数（排除空行和注释行）"""
        code_lines = 0
        in_multiline_comment = False
        
        for line in lines:
            stripped_line = line.strip()
            # 跳过空行
            if not stripped_line:
                continue
            
            # 多行注释处理
            if '"""' in stripped_line or "'''" in stripped_line:
                triple_quotes = '"""' if '"""' in stripped_line else "'''"
                count = stripped_line.count(triple_quotes)
                if count % 2 == 1:
                    in_multiline_comment = not in_multiline_comment
                # 如果是文档字符串（在类/函数定义后的多行字符串），不算作代码行
                continue
            elif in_multiline_comment:
                continue
            # 单行注释
            elif stripped_line.startswith('#'):
                continue
            
            # 有效代码行
            code_lines += 1
        
        return code_lines
    
    def _legacy_calculate_comment_rate(self, lines: List[str]) -> float:
        """计算代码注释率"""
        total_lines = len([line for line in lines if line.strip()])
        if total_lines == 0:
            return 0.0
        
        comment_lines = 0
        in_multiline_comment = False
        
        for line in lines:
            stripped_line = line.strip()
            # 跳过空行
            if not stripped_line:
                continue
            
            # 多行注释
            if '"""' in stripped_line or "'''" in stripped_line:
                triple_quotes = '"""' if '"""' in stripped_line else "'''"
                count = stripped_line.count(triple_quotes)
                if count % 2 == 1:
                    in_multiline_comment = not in_multiline_comment
                comment_lines += 1
            elif in_multiline_comment:
                comment_lines += 1
            # 单行注释
            elif stripped_line.startswith('#'):
                comment_lines += 1
            # 行尾注释
            elif '#' in stripped_line and not stripped_line.startswith('"') and not stripped_line.startswith("'"):
                # 简单判断，避免字符串中的#被误判
                parts = stripped_line.split('#')
                is_comment = True
                for i, part in enumerate(parts[:-1]):
                    if part.count('"') % 2 == 1 or part.count("'") % 2 == 1:
                        is_comment = False
                        break
                if is_comment:
                    comment_lines += 0.5  # 行尾注释算0.5行
        
        return round(comment_lines / total_lines * 100, 2)
    
    def _calculate_cyclomatic_complexity(self, code: str) -> float:
        """使用AST分析计算圈复杂度（精确方法）"""
        tree = ast.parse(code)
        
        # 初始化圈复杂度为1（基本路径）
        cc = 1
        
        # 遍历AST节点，统计决策点
        for node in ast.walk(tree):
            # 条件语句
            if isinstance(node, ast.If):
                cc += 1
            # 循环语句
            elif isinstance(node, ast.For) or isinstance(node, ast.While):
                cc += 1
            # 异常处理
            elif isinstance(node, ast.Try):
                # 每个except分支都是一个决策点
                cc += len(node.handlers)
            # 三元表达式
            elif isinstance(node, ast.IfExp):
                cc += 1
            # 布尔运算符（and, or）
            elif isinstance(node, ast.BoolOp):
                # 每个运算符增加一个分支
                cc += len(node.values) - 1
        
        return float(cc)
    
    def _calculate_nesting_depth(self, code: str) -> int:
        """计算最大嵌套深度（基于缩进的精确方法）"""
        max_depth = 0
        
        for line in code.split('\n'):
            stripped = line.lstrip()
            if not stripped or stripped.startswith('#'):
                continue
            
            indent = len(line) - len(stripped)
            depth = indent // 4
            
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _calculate_fan_out(self, code: str) -> int:
        """使用AST分析计算扇出数（精确方法）"""
        tree = ast.parse(code)
        
        # 提取当前文件定义的函数和类方法
        defined_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                defined_names.add(node.name)
            elif isinstance(node, ast.ClassDef):
                defined_names.add(node.name)
                # 提取类方法
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        defined_names.add(child.name)
        
        # 提取所有函数调用
        external_calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    # 直接函数调用
                    func_name = node.func.id
                    if func_name not in defined_names:
                        external_calls.add(func_name)
                elif isinstance(node.func, ast.Attribute):
                    # 方法调用 obj.method()
                    # 只计算外部库的方法调用
                    attr_name = node.func.attr
                    external_calls.add(attr_name)
        
        # 提取导入的模块
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
        
        # 扇出数 = 外部函数调用数 + 导入模块数
        fan_out = len(external_calls) + len(imports)
        
        return fan_out
    
    def _legacy_calculate_complexity_metrics(self, code: str) -> Dict[str, float]:
        """基于 AST 的复杂度分级，减少字符串模式带来的误判。"""
        tree = self._get_ast_tree(code)

        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.current_loop_depth = 0
                self.max_loop_depth = 0
                self.has_recursion = False
                self.function_stack = []
                self.dynamic_allocations = 0
                self.container_nodes = 0

            def visit_FunctionDef(self, node):
                self.function_stack.append(node.name)
                self.generic_visit(node)
                self.function_stack.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and self.function_stack and node.func.id == self.function_stack[-1]:
                    self.has_recursion = True
                if isinstance(node.func, ast.Name) and node.func.id in {'list', 'dict', 'set', 'tuple'}:
                    self.dynamic_allocations += 1
                if isinstance(node.func, ast.Attribute) and node.func.attr in {'append', 'extend', 'insert', 'copy'}:
                    self.dynamic_allocations += 1
                self.generic_visit(node)

            def visit_For(self, node):
                self.current_loop_depth += 1
                self.max_loop_depth = max(self.max_loop_depth, self.current_loop_depth)
                self.generic_visit(node)
                self.current_loop_depth -= 1

            visit_AsyncFor = visit_For
            visit_While = visit_For

            def visit_List(self, node):
                self.container_nodes += 1
                self.generic_visit(node)

            visit_Set = visit_List
            visit_Dict = visit_List
            visit_ListComp = visit_List
            visit_SetComp = visit_List
            visit_DictComp = visit_List

        visitor = ComplexityVisitor()
        visitor.visit(tree)

        if visitor.has_recursion and visitor.max_loop_depth >= 1:
            time_complexity = 6
        elif visitor.max_loop_depth >= 3:
            time_complexity = 6
        elif visitor.max_loop_depth == 2:
            time_complexity = 5
        elif visitor.max_loop_depth == 1 or visitor.has_recursion:
            time_complexity = 3
        else:
            time_complexity = 1

        if visitor.container_nodes >= 5:
            space_complexity = 5
        elif visitor.container_nodes >= 2 or visitor.has_recursion:
            space_complexity = 3
        elif visitor.container_nodes >= 1 or visitor.dynamic_allocations >= 1:
            space_complexity = 2
        else:
            space_complexity = 1

        complexity_labels = {
            1: 'O(1)',
            2: 'O(log n)',
            3: 'O(n)',
            4: 'O(n log n)',
            5: 'O(n²)',
            6: 'O(n³)',
            7: 'O(2^n)'
        }

        return {
            'time_complexity': float(time_complexity),
            'time_complexity_label': complexity_labels[time_complexity],
            'space_complexity': float(space_complexity),
            'space_complexity_label': complexity_labels[space_complexity]
        }

    def _calculate_code_lines(self, code_or_lines) -> int:
        """基于 tokenize 统计逻辑代码行。"""
        code = code_or_lines if isinstance(code_or_lines, str) else '\n'.join(code_or_lines)
        return self._get_logical_loc(code)

    def _calculate_comment_rate(self, code_or_lines) -> float:
        """基于 tokenize 和 AST 统计注释率。"""
        code = code_or_lines if isinstance(code_or_lines, str) else '\n'.join(code_or_lines)
        logical_loc = self._get_logical_loc(code)
        if logical_loc == 0:
            return 0.0

        comment_lines = self._get_comment_lines(code) + self._get_docstring_lines(code)
        return round(comment_lines / logical_loc * 100, 2)

    def _legacy_calculate_cyclomatic_complexity(self, code: str) -> float:
        """优先使用 radon，否则使用 AST 统计圈复杂度。"""
        if cc_visit is not None:
            try:
                blocks = cc_visit(code)
                if blocks:
                    return float(max(block.complexity for block in blocks))
            except Exception:
                pass

        tree = self._get_ast_tree(code)
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.IfExp)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += max(0, len(node.values) - 1)
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                complexity += len(node.generators)
            elif hasattr(ast, 'Match') and isinstance(node, ast.Match):
                complexity += max(0, len(node.cases) - 1)
        return float(complexity)

    def _legacy_calculate_nesting_depth(self, code: str) -> int:
        """基于 AST 计算真实语法嵌套深度。"""
        tree = self._get_ast_tree(code)
        nesting_nodes = (
            ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With,
            ast.AsyncWith, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef
        )

        def visit(node: ast.AST, depth: int) -> int:
            next_depth = depth + 1 if isinstance(node, nesting_nodes) else depth
            max_depth = next_depth
            for child in ast.iter_child_nodes(node):
                max_depth = max(max_depth, visit(child, next_depth))
            return max_depth

        return max(0, visit(tree, -1))

    def _legacy_calculate_fan_out(self, code: str) -> int:
        """基于 AST 统计导入模块和外部调用目标。"""
        tree = self._get_ast_tree(code)
        defined_functions = set()
        imported_aliases = set()
        imported_modules = set()

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined_functions.add(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name.split('.')[0])
                    imported_aliases.add(alias.asname or alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module.split('.')[0])
                for alias in node.names:
                    imported_aliases.add(alias.asname or alias.name)

        external_calls = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                if node.func.id in imported_aliases:
                    external_calls.add(node.func.id)
                elif node.func.id not in defined_functions and node.func.id not in dir(__builtins__):
                    external_calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                root = node.func.value
                if isinstance(root, ast.Name) and root.id in imported_aliases:
                    external_calls.add(f"{root.id}.{node.func.attr}")

        return len(imported_modules | external_calls)

    def _legacy_calculate_halstead_metrics(self, code: str) -> Dict[str, float]:
        """优先使用第三方库，否则基于 tokenize 计算 Halstead 指标。"""
        if h_visit is not None:
            try:
                report = h_visit(code)
                total = report.total
                bugs = total.volume / 3000 if total.volume else 0.0
                return {
                    'halstead_n1': round(total.h1, 2),
                    'halstead_n2': round(total.h2, 2),
                    'halstead_N1': round(total.N1, 2),
                    'halstead_N2': round(total.N2, 2),
                    'halstead_vocabulary': round(total.vocabulary, 2),
                    'halstead_length': round(total.length, 2),
                    'halstead_volume': round(total.volume, 2),
                    'halstead_difficulty': round(total.difficulty, 2),
                    'halstead_effort': round(total.effort, 2),
                    'halstead_time': round(total.time, 2),
                    'halstead_bugs': round(bugs, 4)
                }
            except Exception:
                pass

        operators, operands = self._get_operator_and_operand_tokens(code)
        n1 = len(set(operators))
        n2 = len(set(operands))
        N1 = len(operators)
        N2 = len(operands)
        n = n1 + n2
        N = N1 + N2
        volume = N * math.log2(n) if n > 1 else 0.0
        difficulty = (n1 / 2) * (N2 / n2) if n1 > 0 and n2 > 0 else 0.0
        effort = difficulty * volume
        time_required = effort / 18 if effort > 0 else 0.0
        bugs = volume / 3000 if volume > 0 else 0.0

        return {
            'halstead_n1': round(n1, 2),
            'halstead_n2': round(n2, 2),
            'halstead_N1': round(N1, 2),
            'halstead_N2': round(N2, 2),
            'halstead_vocabulary': round(n, 2),
            'halstead_length': round(N, 2),
            'halstead_volume': round(volume, 2),
            'halstead_difficulty': round(difficulty, 2),
            'halstead_effort': round(effort, 2),
            'halstead_time': round(time_required, 2),
            'halstead_bugs': round(bugs, 4)
        }

    def _legacy_calculate_maintainability_index(self, volume: float, cyclomatic_complexity: float, code_lines: int) -> float:
        """使用标准公式计算维护性指数。"""
        if volume <= 0 or code_lines <= 0:
            return 0.0

        mi = 171 - 5.2 * math.log(volume) - 0.23 * cyclomatic_complexity - 16.2 * math.log(code_lines)
        mi = max(0.0, min(100.0, mi * 100 / 171))
        return round(mi, 2)

    def _generate_optimization_suggestions(self, metrics: Dict[str, Any], parsed_data: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        suggestions = []
        
        # 检查各项指标是否达标
        if not metrics.get('code_lines_passed', False):
            suggestions.append(f"代码行数 {metrics['code_lines']} 行未达标：建议精简代码，提取重复逻辑为函数，使用更简洁的实现方式")
        
        if not metrics.get('comment_rate_passed', False):
            suggestions.append(f"注释率 {metrics['comment_rate']}% 未达标：建议在核心函数（如fit、predict）中添加详细的功能说明和参数文档")
        
        if not metrics.get('cyclomatic_complexity_passed', False):
            suggestions.append(f"圈复杂度 {metrics['cyclomatic_complexity']} 未达标：建议将复杂函数拆分为多个子函数，减少嵌套条件判断")
        
        if not metrics.get('nesting_depth_passed', False):
            suggestions.append(f"嵌套深度 {metrics['nesting_depth']} 未达标：建议减少if/loop嵌套层级，使用提前返回或函数拆分优化")
        
        if not metrics.get('fan_out_passed', False):
            suggestions.append(f"扇出数 {metrics['fan_out']} 未达标：建议减少外部依赖，合并相似功能，降低模块间耦合度")
        
        if not metrics.get('test_pass_rate_passed', False):
            test_pass_rate = metrics.get('test_pass_rate', 0)
            suggestions.append(f"测试用例通过率 {test_pass_rate}% 未达标：建议检查代码逻辑，修复bug，确保所有测试用例通过")
        
        return suggestions
    
    def _calculate_overall_rating(self, metrics: Dict[str, Any]) -> str:
        """计算综合评价"""
        # 统计达标指标数量
        passed_count = 0
        for metric_name in self.target_thresholds.keys():
            if metrics.get(f'{metric_name}_passed', False):
                passed_count += 1
        
        total_metrics = len(self.target_thresholds)
        pass_rate = passed_count / total_metrics
        
        if pass_rate >= 0.95:
            return "优秀"
        elif pass_rate >= 0.85:
            return "良好"
        elif pass_rate >= 0.7:
            return "合格"
        else:
            return "待优化"
    
    def execute_tests(self, code_content: str, test_content: str, entry_point: str = None) -> float:
        """
        执行测试用例，计算通过率
        
        Args:
            code_content: 代码内容
            test_content: 测试用例内容
            entry_point: 函数入口点
            
        Returns:
            测试用例通过率（百分比）
        """
        # 清理测试用例内容，替换可能的转义字符
        test_content = test_content.replace('\\n', '\n')
        
        # 创建一个临时命名空间
        namespace = {}
        
        try:
            # 执行代码，将所有内容添加到命名空间
            exec(code_content, namespace)
            
            # 定义 candidate 函数，根据 entry_point 调用相应的函数
            if entry_point:
                # 解析 entry_point，支持 'ClassName.method' 或 'ClassName().method' 或直接 'function_name'
                if '.' in entry_point:
                    # 处理 ClassName().method 或 ClassName.method 格式
                    parts = entry_point.split('.')
                    if len(parts) == 2:
                        class_name, method_name = parts
                        # 移除可能的括号
                        class_name = class_name.replace('()', '')
                        
                        def candidate(*args, **kwargs):
                            if class_name in namespace:
                                instance = namespace[class_name]()
                                if hasattr(instance, method_name):
                                    return getattr(instance, method_name)(*args, **kwargs)
                            return None
                    else:
                        # 如果格式不对，返回 None
                        def candidate(*args, **kwargs):
                            return None
                else:
                    def candidate(*args, **kwargs):
                        if entry_point in namespace:
                            return namespace[entry_point](*args, **kwargs)
                        return None
            else:
                # 自动检测代码中的入口函数
                entry_func_names = []
                try:
                    tree = ast.parse(code_content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            entry_func_names.append(node.name)
                except:
                    pass
                
                if entry_func_names:
                    entry_func_name = entry_func_names[0]
                    def candidate(*args, **kwargs):
                        if entry_func_name in namespace:
                            return namespace[entry_func_name](*args, **kwargs)
                        return None
                else:
                    # 默认尝试 Solution 类的方法（兼容旧代码）
                    def candidate(*args, **kwargs):
                        if 'Solution' in namespace:
                            instance = namespace['Solution']()
                            for method_name in ['twoSum', 'solve', 'run', 'execute']:
                                if hasattr(instance, method_name):
                                    return getattr(instance, method_name)(*args, **kwargs)
                        return None
            
            # 将 candidate 函数添加到命名空间
            namespace['candidate'] = candidate
            
            # 检查测试用例格式：是否是 def check(candidate): 格式
            if 'def check(candidate):' in test_content or test_content.strip().startswith('def check'):
                # 执行整个测试函数定义
                exec(test_content, namespace)
                
                # 调用 check 函数来执行所有测试
                if 'check' in namespace:
                    try:
                        # 使用超时机制执行check函数
                        def run_check():
                            namespace['check'](candidate)
                        
                        run_with_timeout(run_check, 10)  # 10秒超时
                        # 如果所有测试通过，返回 100%
                        return 100.0
                    except TimeoutException:
                        print("测试执行超时，跳过")
                        return 0.0
                    except AssertionError:
                        # 测试失败，需要逐个执行来统计通过率
                        pass
                
                # 提取单个 assert 语句来统计通过率
                # 从 test_content 中提取 assert 语句
                test_cases = []
                lines = test_content.split('\n')
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('assert candidate('):
                        test_cases.append(stripped)
                
                if not test_cases:
                    return 0.0
                
                # 执行测试用例
                passed = 0
                total = len(test_cases)
                
                for test_case in test_cases:
                    try:
                        # 使用超时机制执行测试用例
                        def run_test_case():
                            exec(test_case, namespace)
                        
                        run_with_timeout(run_test_case, 10)  # 10秒超时
                        passed += 1
                    except TimeoutException:
                        # 测试超时，跳过
                        print("测试用例执行超时，跳过")
                        pass
                    except AssertionError:
                        # 测试失败
                        pass
                    except Exception as e:
                        # 其他错误
                        pass
                
                # 计算通过率
                pass_rate = (passed / total) * 100
                return round(pass_rate, 2)
            else:
                # 旧的测试用例格式：直接使用 assert 语句
                # 提取测试用例
                test_cases = []
                test_cases = test_content.split('    assert ')
                test_cases = [f'assert {tc.strip()}' for tc in test_cases if tc.strip()]
                
                # 如果没有提取到测试用例，尝试逐行提取
                if not test_cases:
                    lines = test_content.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith('assert candidate('):
                            test_cases.append(line)
                
                if not test_cases:
                    return 0.0
                
                # 执行测试用例
                passed = 0
                total = len(test_cases)
                
                for test_case in test_cases:
                    try:
                        # 使用超时机制执行测试用例
                        def run_test_case():
                            exec(test_case, namespace)
                        
                        run_with_timeout(run_test_case, 10)  # 10秒超时
                        passed += 1
                    except TimeoutException:
                        # 测试超时，跳过
                        print("测试用例执行超时，跳过")
                        pass
                    except AssertionError:
                        # 测试失败
                        pass
                    except Exception as e:
                        # 其他错误
                        pass
                
                # 计算通过率
                pass_rate = (passed / total) * 100
                return round(pass_rate, 2)
            
        except Exception as e:
            # 如果执行过程中出现错误，返回 0% 通过率
            return 0.0
    
    def generate_report(self, parsed_data: Dict[str, Any], metrics: Dict[str, Any]) -> str:
        """
        生成评价报告
        
        Args:
            parsed_data: 解析数据
            metrics: 指标数据
            
        Returns:
            格式化的报告字符串
        """
        report = []
        
        # 指标行（基础代码质量指标 + Halstead 指标 + 复杂度指标）
        metric_names = {
            'code_lines': '代码行数',
            'cyclomatic_complexity': '圈复杂度',
            'comment_rate': '代码注释率(%)',
            'nesting_depth': '嵌套深度(层)',
            'fan_out': '扇出数',
            'test_pass_rate': '测试用例通过率(%)',
            # Halstead 指标
            'halstead_vocabulary': 'Halstead词典大小',
            'halstead_length': 'Halstead程序长度',
            'halstead_volume': 'Halstead程序容量',
            'halstead_difficulty': 'Halstead难度',
            'halstead_effort': 'Halstead工作量',
            'halstead_time': 'Halstead时间(秒)',
            'halstead_bugs': 'Halstead错误数',
            # 可维护性指数
            'maintainability_index': '可维护性指数MI(%)',
            # 复杂度指标
            'time_complexity': '时间复杂度等级',
            'time_complexity_label': '时间复杂度',
            'space_complexity': '空间复杂度等级',
            'space_complexity_label': '空间复杂度'
        }
        
        for metric_key, display_name in metric_names.items():
            if metric_key not in metrics:
                continue
            
            value = metrics[metric_key]
            # 格式化结果行：指标名称:实际结果
            if isinstance(value, (int, float)):
                row = f"{display_name}:{value:.2f}"
            else:
                row = f"{display_name}:{value}"
            report.append(row)
        
        return '\n'.join(report)
    
    def evaluate(self, file_path: str, test_pass_rate: float = 0.0) -> str:
        """
        执行完整的评价流程
        
        Args:
            file_path: 文件路径
            test_pass_rate: 测试用例通过率
            
        Returns:
            评价报告
        """
        # 1. 文件校验
        valid, message = self.validate_file(file_path)
        if not valid:
            return f"错误: {message}"
        
        # 2. 代码解析
        parsed_data = self.parse_code(file_path)
        
        # 3. 指标计算
        metrics = self.calculate_metrics(parsed_data)
        
        # 4. 添加测试通过率指标
        metrics['test_pass_rate'] = test_pass_rate
        
        # 5. 生成报告
        report = self.generate_report(parsed_data, metrics)
        
        return report


def process_single_row(ws, row_num, answer_index, test_index, quantitative_metric_index, evaluator):
    """
    处理单行数据
    
    Args:
        ws: 工作表对象
        row_num: 行号（从2开始）
        answer_index: answer 列索引
        test_index: test 列索引
        quantitative_metric_index: QuantitativeMetric 列索引
        evaluator: 评估器对象
        
    Returns:
        是否成功处理
    """
    try:
        print(f"开始处理第 {row_num} 行...")
        
        # 检查QuantitativeMetric列是否已有值
        quantitative_metric_value = ws[row_num][quantitative_metric_index].value
        print(f"第 {row_num} 行QuantitativeMetric值: {quantitative_metric_value}")
        if quantitative_metric_value:
            print(f"第 {row_num} 行数据的 QuantitativeMetric 列已有值，跳过")
            return False
        
        # 获取当前行的 answer 列内容
        answer_value = ws[row_num][answer_index].value
        
        if not answer_value:
            print(f"第 {row_num} 行数据的 answer 列为空，跳过")
            return False
        
        # 获取当前行的 test 列内容
        test_value = ws[row_num][test_index].value
        
        if not test_value:
            print(f"第 {row_num} 行数据的 test 列为空，跳过")
            return False
        
        # 处理 answer 列内容，替换 \n 为实际换行符
        code_content = answer_value.replace('\\n', '\n')
        
        # 去除代码块标记（如果有）
        code_content = code_content.strip()
        
        # 检查并移除开头的代码块标记
        if code_content.startswith('```python'):
            code_content = code_content[9:]
        elif code_content.startswith('```'):
            code_content = code_content[3:]
        elif code_content.startswith('``python'):
            code_content = code_content[8:]
        elif code_content.startswith('``'):
            code_content = code_content[2:]
        
        # 检查并移除结尾的代码块标记
        code_content = code_content.strip()
        if code_content.endswith('```'):
            code_content = code_content[:-3]
        elif code_content.endswith('``'):
            code_content = code_content[:-2]
        
        code_content = code_content.strip()
        
        # 创建临时文件来存储代码（只使用answer列的代码）
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(code_content)
            temp_file_path = temp_file.name
        
        try:
            # 执行测试并计算通过率（使用answer列的代码）
            test_pass_rate = evaluator.execute_tests(code_content, test_value)
            # 评估代码质量指标（使用answer列的代码）
            report = evaluator.evaluate(temp_file_path, test_pass_rate)
            
            # 将报告写入到 QuantitativeMetric 列
            ws.cell(row=row_num, column=quantitative_metric_index + 1, value=report)
            
            print(f"第 {row_num} 行处理完成")
            return True
        finally:
            # 删除临时文件
            try:
                os.unlink(temp_file_path)
            except:
                pass
    except Exception as e:
        print(f"第 {row_num} 行处理出错: {str(e)}")
        return False


def run_quantitative_evaluation(
    excel_file_path,
    answer_column="answer",
    test_column="test",
    output_column="QuantitativeMetric",
    max_rows=440
):
    """主函数"""
    print(f"处理文件: {excel_file_path}")
    print(f"输入列: {answer_column}, 测试列: {test_column}, 输出列: {output_column}")
    print("=" * 80)
    
    try:
        # 读取 Excel 文件
        wb = openpyxl.load_workbook(excel_file_path)
        ws = wb.active
        
        # 获取列名
        headers = [cell.value for cell in ws[1]]
        
        # 检查是否存在必要的列
        if answer_column not in headers:
            print(f"错误: Excel 文件中未找到 {answer_column} 列")
            return
        
        if test_column not in headers:
            print(f"错误: Excel 文件中未找到 {test_column} 列")
            return
        
        # 检查是否存在 output_column 列，如果不存在则创建
        if output_column not in headers:
            # 在最后一列后添加 output_column 列
            ws.cell(row=1, column=len(headers) + 1, value=output_column)
            headers.append(output_column)
            print(f"已添加 {output_column} 列")
        
        # 获取列索引
        answer_index = headers.index(answer_column)
        test_index = headers.index(test_column)
        quantitative_metric_index = headers.index(output_column)
        
        # 创建评估器
        evaluator = MLCodeEvaluator()
        
        # 获取数据总行数
        total_rows = ws.max_row
        print(f"Excel 文件共有 {total_rows - 1} 条数据需要处理")
        print("=" * 80)
        
        # 处理每一行数据（从第2行开始，第1行是表头）
        success_count = 0
        save_count = 0
        max_rows_to_process = min(max_rows, total_rows - 1)
        print(f"将处理前 {max_rows_to_process} 条数据")
        for row_num in range(2, 2 + max_rows_to_process):
            success = process_single_row(ws, row_num, answer_index, test_index, quantitative_metric_index, evaluator)
            if success:
                success_count += 1
                save_count += 1
                
                # 每成功写入20条保存一次文件（减少保存频率，避免文件损坏）
                if save_count % 20 == 0:
                    print(f"保存文件中... (已成功写入 {save_count} 条)")
                    try:
                        wb.save(excel_file_path)
                        print("保存成功")
                    except Exception as e:
                        print(f"保存文件时出错: {e}")
        
        # 最终保存 Excel 文件
        print("执行最终保存...")
        try:
            wb.save(excel_file_path)
            print("最终保存成功")
        except Exception as e:
            print(f"最终保存文件时出错: {e}")
        
        print("=" * 80)
        print(f"处理完成！共处理 {max_rows_to_process} 条数据，成功 {success_count} 条")
        print(f"报告已写入到 {output_column} 列")
    
    except Exception as e:
        print(f"错误: {str(e)}")


if __name__ == "__main__":
    run_quantitative_evaluation(
        excel_file_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'runs', 'working.xlsx'),
        answer_column="answer",
        test_column="test",
        output_column="QuantitativeMetric",
        max_rows=440
    )
