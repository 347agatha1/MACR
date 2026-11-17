"""
多头注意力矩阵融合模块

该模块实现了基于多头注意力的矩阵信息融合机制，
能够将输入矩阵的信息进行充分融合，同时保持输入输出维度一致。
"""

import torch
import torch.nn as nn
import numpy as np
import os


class MultiHeadFusion(nn.Module):
    """
    多头注意力融合模块
    
    该模块使用多头注意力机制对输入矩阵进行信息融合，
    通过多个注意力头学习不同的表示，然后融合结果。
    包含残差连接和层归一化，确保训练稳定性。
    """
    
    def __init__(self, input_dim, num_heads=8, dropout=0.1):
        """
        初始化多头注意力融合模块
        
        Args:
            input_dim: 输入矩阵的维度（列数）
            num_heads: 注意力头的数量，必须能整除input_dim
            dropout: Dropout比率
        """
        super(MultiHeadFusion, self).__init__()
        
        # 验证输入参数
        if input_dim % num_heads != 0:
            raise ValueError(f"input_dim ({input_dim}) 必须能被 num_heads ({num_heads}) 整除")
        
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        # 定义多头注意力层
        # embed_dim设置为input_dim，保证输出维度与输入一致
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # 使用batch first模式
        )
        
        # 前馈网络层
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 4, input_dim)
        )
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入矩阵，形状为 (batch_size, seq_len, input_dim)
            
        Returns:
            融合后的矩阵，形状与输入相同
        """
        # 保存原始输入用于残差连接
        residual = x
        
        # 第一步：多头注意力融合
        # Self-attention: Q, K, V 都使用输入x
        attn_output, _ = self.attention(x, x, x)
        
        # 残差连接和层归一化
        attn_output = self.dropout(attn_output)
        x = self.layer_norm1(residual + attn_output)
        
        # 保存中间结果用于第二个残差连接
        residual2 = x
        
        # 第二步：前馈网络处理
        ff_output = self.feed_forward(x)
        
        # 残差连接和层归一化
        ff_output = self.dropout(ff_output)
        x = self.layer_norm2(residual2 + ff_output)
        
        return x


def load_matrix_from_file(filepath):
    """
    从文件加载矩阵数据
    
    Args:
        filepath: 文件路径
        
    Returns:
        numpy数组格式的矩阵
    """
    matrix = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                row = [int(x) for x in line.strip('[]').split(',')]
                matrix.append(row)
    
    return np.array(matrix)


def matrix_to_tensor(matrix):
    """
    将numpy矩阵转换为PyTorch张量
    
    Args:
        matrix: numpy数组
        
    Returns:
        PyTorch张量，形状为 (1, seq_len, input_dim)
    """
    # 添加batch维度
    tensor = torch.tensor(matrix, dtype=torch.float32)
    return tensor.unsqueeze(0)  # (1, seq_len, input_dim)


def tensor_to_matrix(tensor):
    """
    将PyTorch张量转换回矩阵
    
    Args:
        tensor: PyTorch张量
        
    Returns:
        numpy数组
    """
    # 移除batch维度
    return tensor.squeeze(0).numpy()


def fuse_matrix_with_attention(input_matrix, num_heads=8, dropout=0.1, keep_integer=False):
    """
    使用多头注意力融合矩阵信息
    
    Args:
        input_matrix: 输入矩阵，numpy数组
        num_heads: 注意力头数量
        dropout: Dropout比率
        keep_integer: 是否保持整数输出
        
    Returns:
        融合后的矩阵，numpy数组，维度与输入一致
    """
    # 获取输入维度
    input_dim = input_matrix.shape[1]
    
    # 创建融合模型
    fusion_model = MultiHeadFusion(
        input_dim=input_dim,
        num_heads=num_heads,
        dropout=dropout
    )
    
    # 设置为评估模式
    fusion_model.eval()
    
    # 转换为张量
    input_tensor = matrix_to_tensor(input_matrix)
    
    # 执行融合
    with torch.no_grad():
        output_tensor = fusion_model(input_tensor)
    
    # 转换回numpy数组
    output_matrix = tensor_to_matrix(output_tensor)
    
    if keep_integer:
        # 四舍五入到最近整数
        output_matrix = np.round(output_matrix).astype(int)
        # 限制范围到原始数据的合理范围（如0-10）
        output_matrix = np.clip(output_matrix, 0, 10)
    
    return output_matrix


def demo_multi_head_fusion():
    """
    演示多头注意力融合功能
    """
    # 获取当前脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "QualitativeMatrix.txt")
    
    # 加载示例矩阵数据
    print("加载输入矩阵...")
    input_matrix = load_matrix_from_file(data_file)
    
    print(f"输入矩阵形状: {input_matrix.shape}")
    print("输入矩阵:")
    print(input_matrix)
    
    # 使用多头注意力进行融合
    print("\n执行多头注意力融合...")
    output_matrix = fuse_matrix_with_attention(
        input_matrix, 
        num_heads=2,
        dropout=0.0
    )
    
    print(f"输出矩阵形状: {output_matrix.shape}")
    print("输出矩阵:")
    print(output_matrix)
    
    # 维度一致性检查
    print("\n维度一致性检查:")
    print(f"输入维度: {input_matrix.shape}")
    print(f"输出维度: {output_matrix.shape}")
    print(f"维度是否一致: {input_matrix.shape == output_matrix.shape}")
    
    # 计算融合前后的变化
    difference = np.abs(output_matrix - input_matrix)
    print(f"\n融合前后差异统计:")
    print(f"最大差异: {difference.max():.4f}")
    print(f"平均差异: {difference.mean():.4f}")
    print(f"标准差: {difference.std():.4f}")


if __name__ == "__main__":
    demo_multi_head_fusion()

