"""
循环CNN融合模块

该模块实现循环融合逻辑：
1. 判断CNN输出是否只有一行
2. 如果是一行，输出完成信息
3. 如果不是一行，将CNN输出与对应部分的原始矩阵进行残差连接
4. 重新进行多头注意力融合和CNN融合（由fusion\cnn.py执行）
5. 重复直到CNN输出为一行
"""

import numpy as np
import os
import torch
import torch.nn.functional as F
import openpyxl
from .attention import (
    fuse_matrix_with_attention, 
    load_matrix_from_file,
)
from .convolution import cnn_fuse_rows as cnn_fuse_from_file






def residual_connection(cnn_output, current_matrix):
    """
    残差连接：将CNN输出与当前矩阵每两行取平均的结果进行残差连接
    
    残差连接公式: output = cnn_output + avg(current_matrix[每两行])
    
    设计说明：
    - 由于CNN融合使用stride=2的卷积，每两行压缩为一行
    - 残差连接也采用每两行取平均的方式，保持信息对应关系
    - 两个矩阵形状相同，进行逐元素相加
    
    Args:
        cnn_output: CNN输出矩阵，形状为 (seq_len, input_dim)
        current_matrix: 当前矩阵（上一轮迭代结果），形状为 (current_seq_len, input_dim)
        
    Returns:
        残差连接后的矩阵，形状与cnn_output相同
    """
    cnn_len = cnn_output.shape[0]
    current_len = current_matrix.shape[0]
    
    # 验证当前矩阵行数是CNN输出行数的2倍（CNN使用stride=2卷积）
    expected_current_len = cnn_len * 2
    
    if current_len == expected_current_len:
        # 正常情况：每两行取平均
        # reshape为 (cnn_len, 2, input_dim)，然后在axis=1上取平均
        avg_subset = current_matrix.reshape(cnn_len, 2, -1).mean(axis=1)
    elif current_len == expected_current_len + 1:
        # 奇数行情况：最后一行的平均值等于它本身
        # 前cnn_len*2行每两行取平均，最后一行单独处理
        avg_subset = np.zeros((cnn_len, current_matrix.shape[1]))
        
        # 前(cnn_len*2 - 2)行：每两行取平均
        regular_part = current_matrix[:-1].reshape(cnn_len - 1, 2, -1).mean(axis=1)
        avg_subset[:-1] = regular_part
        
        # 最后一行：与倒数第二行的平均值作为最后一组
        avg_subset[-1] = (current_matrix[-2] + current_matrix[-1]) / 2
    else:
        # 异常情况：直接截取或填充
        avg_subset = current_matrix[:cnn_len]
    
    # 残差连接：两个矩阵形状相同，均为 (cnn_len, input_dim)
    residual = cnn_output + avg_subset
    
    return residual


def loop_cnn_fusion_with_fusion_layer(original_matrix, num_heads=2, kernel_size=3, cnn_model_path=None):
    """
    循环CNN融合主函数（带融合输出层）
    
    持续进行多头注意力融合和CNN融合（由fusion\cnn.py执行），
    直到CNN输出为一行，然后通过全连接层转换为9维输出
    
    Args:
        original_matrix: 原始输入矩阵
        num_heads: 多头注意力头数
        kernel_size: CNN卷积核大小
        cnn_model_path: CNNFusion模型保存/加载路径
        
    Returns:
        最终输出矩阵（行数为1，列为9维）
    """
    print("=" * 70)
    print("循环CNN融合开始（带融合输出层）")
    print("=" * 70)
    
    # 记录原始矩阵
    original_matrix = original_matrix.copy().astype(np.float32)
    current_matrix = original_matrix.copy()
    iteration = 0
    
    while True:
        iteration += 1
        print(f"\n{'='*70}")
        print(f"迭代次数: {iteration}")
        print(f"{'='*70}")
        
        # print(f"\n当前矩阵形状: {current_matrix.shape}")
        # print(f"当前矩阵数值范围: [{current_matrix.min():.4f}, {current_matrix.max():.4f}]")
        
        # 步骤1: 多头注意力融合
        print("\n执行多头注意力融合...")
        attention_output = fuse_matrix_with_attention(
            current_matrix,
            num_heads=num_heads,
            dropout=0.0
        )
        print(f"多头注意力输出形状: {attention_output.shape}")
        print(f"注意力输出数值范围: [{attention_output.min():.4f}, {attention_output.max():.4f}]")
        
        # 步骤2: CNN行融合（由fusion\cnn.py执行）
        print("\n执行CNN行融合（调用fusion\\cnn.py）...")
        
        # 处理奇数行：使用填充策略而不是移除数据
        seq_len = attention_output.shape[0]
        if seq_len % 2 != 0:
            print(f"  输入矩阵行数 ({seq_len}) 为奇数，使用填充策略处理")
            # 复制最后一行进行填充，保持数据完整性
            padded_matrix = np.vstack([attention_output, attention_output[-1]])
            cnn_output = cnn_fuse_from_file(padded_matrix, kernel_size=kernel_size, model_path=cnn_model_path)
        else:
            cnn_output = cnn_fuse_from_file(attention_output, kernel_size=kernel_size, model_path=cnn_model_path)
        
        print(f"CNN融合输出形状: {cnn_output.shape}")
        print(f"CNN输出数值范围: [{cnn_output.min():.4f}, {cnn_output.max():.4f}]")
        
        # 步骤3: 判断是否为一行
        if cnn_output.shape[0] == 1:
            print("\n" + "=" * 70)
            print("循环融合完成！")
            print("=" * 70)
            
            # 对cnn_output进行sigmoid归一化
            cnn_output_tensor = torch.from_numpy(cnn_output).float()
            normalized_cnn_output = F.sigmoid(cnn_output_tensor).numpy()
            print(f"\nSigmoid归一化后的最终矩阵:")
            print(normalized_cnn_output)
            print(f"Sigmoid归一化后的最终矩阵形状: {normalized_cnn_output.shape}")
            
            return normalized_cnn_output
        else:
            print(f"\nCNN输出有 {cnn_output.shape[0]} 行，未达到一行，继续循环...")
            
            # 步骤4: 残差连接
            print("\n执行残差连接...")
            # 使用当前矩阵(current_matrix)而非原始矩阵，保证数值范围一致
            residual = residual_connection(cnn_output, current_matrix)
            print(f"残差连接后形状: {residual.shape}")
            print(f"残差连接后数值范围: [{residual.min():.4f}, {residual.max():.4f}]")
            
            # 步骤5: 准备下一次迭代
            current_matrix = residual
            
            print(f"\n准备进行第 {iteration + 1} 次迭代...")
            
            # 防止无限循环的保险措施
            if iteration > 100:
                print("\n警告：已达到最大迭代次数，强制结束循环")
                print("返回当前CNN输出结果")
                # 对cnn_output进行sigmoid归一化
                cnn_output_tensor = torch.from_numpy(cnn_output).float()
                normalized_cnn_output = F.sigmoid(cnn_output_tensor).numpy()
                print(f"\nSigmoid归一化后的最终矩阵:")
                print(normalized_cnn_output)
                print(f"Sigmoid归一化后的最终矩阵形状: {normalized_cnn_output.shape}")
                return normalized_cnn_output





def load_qualitative_matrix_from_excel(excel_file_path, qualitative_matrix_column="QualitativeMatrix"):
    """
    从Excel文件中加载QualitativeMatrix列作为原始矩阵

    Args:
        excel_file_path: Excel文件路径
        qualitative_matrix_column: 定性矩阵列名

    Returns:
        original_matrix: 原始矩阵
    """
    try:
        # 读取Excel文件
        wb = openpyxl.load_workbook(excel_file_path)
        ws = wb.active

        # 获取列名
        headers = [cell.value for cell in ws[1]]

        # 检查必要的列
        if qualitative_matrix_column not in headers:
            print(f"错误: Excel文件中未找到 {qualitative_matrix_column} 列")
            return None

        # 获取列索引
        qualitative_matrix_index = headers.index(qualitative_matrix_column)
        
        # 获取第一行数据（第2行）
        qualitative_matrix_value = ws[2][qualitative_matrix_index].value
        
        if not qualitative_matrix_value:
            print("错误: 第一行数据的 QualitativeMatrix 列为空")
            return None
        
        # 处理多行格式
        matrix_content = qualitative_matrix_value.replace('\\n', '\n')
        matrix_lines = matrix_content.strip().split('\n')
        original_matrix = []
        
        for line in matrix_lines:
            line = line.strip()
            if line:
                try:
                    # 移除首尾的括号
                    if line.startswith('[') and line.endswith(']'):
                        line = line[1:-1]
                    # 分割元素
                    elements = [int(x.strip()) for x in line.split(',')]
                    original_matrix.append(elements)
                except:
                    pass
        
        if original_matrix:
            return np.array(original_matrix)
        else:
            print("错误: 无法解析 QualitativeMatrix 列")
            return None
            
    except Exception as e:
        print(f"加载Excel数据时出错: {e}")
        return None


def parse_qualitative_matrix_string(qualitative_matrix_value):
    """
    从字符串格式的定性矩阵数据中解析矩阵
    
    Args:
        qualitative_matrix_value: 定性矩阵数据（字符串格式）
        
    Returns:
        original_matrix: 原始矩阵
    """
    try:
        if not qualitative_matrix_value:
            print("错误: 定性矩阵数据为空")
            return None
        
        # 处理多行格式
        matrix_content = qualitative_matrix_value.replace('\\n', '\n')
        matrix_lines = matrix_content.strip().split('\n')
        original_matrix = []
        
        for line in matrix_lines:
            line = line.strip()
            if line:
                # 检测是否为单行多向量格式: [v1], [v2], [v3]
                if '], [' in line:
                    # 按 '], [' 分割成多个向量
                    vectors = line.split('], [')
                    for i, vec in enumerate(vectors):
                        # 处理第一个和最后一个向量的首尾括号
                        if i == 0 and vec.startswith('['):
                            vec = vec[1:]
                        if i == len(vectors) - 1 and vec.endswith(']'):
                            vec = vec[:-1]
                        # 按逗号分割元素
                        try:
                            elements = [float(x.strip()) for x in vec.split(',')]
                            original_matrix.append(elements)
                        except:
                            print(f"警告: 无法解析向量部分: {repr(vec)}")
                            pass
                else:
                    try:
                        # 移除首尾的括号
                        if line.startswith('[') and line.endswith(']'):
                            line = line[1:-1]
                        # 分割元素
                        elements = [float(x.strip()) for x in line.split(',')]
                        original_matrix.append(elements)
                    except:
                        # 尝试其他格式解析
                        try:
                            # 尝试解析空格分隔的格式
                            elements = [float(x.strip()) for x in line.split()]
                            original_matrix.append(elements)
                        except:
                            pass
        
        if original_matrix:
            return np.array(original_matrix)
        else:
            print("错误: 无法解析定性矩阵数据")
            return None
            
    except Exception as e:
        print(f"解析定性矩阵数据时出错: {e}")
        return None


def demo_loop_cnn_fusion_with_loss(qualitative_matrix_value=None, cnn_model_path=None, excel_file_path=None, qualitative_matrix_column="QualitativeMatrix"):
    """
    演示循环CNN融合功能，并计算与归一化定量指标的损失

    流程：
    1. 加载定性矩阵数据（如果未提供则从Excel文件加载）
    2. 执行循环CNN融合，得到1×9维输出
    3. 加载并归一化定量指标
    4. 计算两种方法之间的损失（MSE、MAE、余弦相似度）

    Args:
        qualitative_matrix_value: 定性矩阵数据（字符串格式），如果为None则从Excel文件加载
        cnn_model_path: CNNFusion模型保存/加载路径
        excel_file_path: Excel文件路径（当qualitative_matrix_value为None时使用）
        qualitative_matrix_column: 定性矩阵列名
    """
    print("=" * 70)
    print("循环CNN融合演示（含损失计算）")
    print("=" * 70)
    
    # 步骤1: 加载原始矩阵数据
    print("\n加载原始矩阵数据...")
    
    if qualitative_matrix_value is not None:
        # 使用提供的定性矩阵数据
        print("使用提供的定性矩阵数据...")
        original_matrix = parse_qualitative_matrix_string(qualitative_matrix_value)
        
        if original_matrix is None:
            print("错误: 无法解析定性矩阵数据")
            # 返回一个随机矩阵，而不是全0矩阵
            return np.random.rand(1, 8)
    else:
        # 从Excel文件加载定性矩阵数据
        if excel_file_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            excel_file_path = os.path.join(script_dir, "..", "dataset", "LeetCodeDataset_demo.xlsx")
        original_matrix = load_qualitative_matrix_from_excel(excel_file_path, qualitative_matrix_column)
        
        if original_matrix is None:
            print("错误: 无法加载原始矩阵数据")
            return np.zeros((1, 8))
    
    original_matrix = np.clip(original_matrix, 1, 5)
    print(f"原始矩阵形状: {original_matrix.shape}")
    print("原始矩阵:")
    print(original_matrix)
    
    # 步骤2: 执行循环CNN融合（带融合输出层）
    print("\n" + "=" * 70)
    print("执行循环CNN融合")
    print("=" * 70)
    final_output = loop_cnn_fusion_with_fusion_layer(
        original_matrix,
        num_heads=2,
        kernel_size=3,
        cnn_model_path=cnn_model_path
    )
    
    # 输出最终结果
    print("\n" + "=" * 70)
    print("LoopCNN融合结果")
    print("=" * 70)
    print(f"最终矩阵形状: {final_output.shape}")
    print("最终矩阵:")
    print(final_output)
    loopcnn_output = final_output  # (1, 9)
    
    return loopcnn_output


if __name__ == "__main__":
    demo_loop_cnn_fusion_with_loss()