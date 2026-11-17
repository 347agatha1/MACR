"""
CNN矩阵融合模块

该模块接收多头注意力融合的输出，将每两行数据使用1D卷积进行融合，
最终将矩阵高度减半，宽度保持不变。
"""

import torch
import torch.nn as nn
import numpy as np
import os
from .attention import fuse_matrix_with_attention, load_matrix_from_file


class CNNFusion(nn.Module):
    """
    CNN行融合模块
    
    使用1D卷积将相邻两行融合为一行，
    实现矩阵高度减半，宽度保持不变。
    """
    
    def __init__(self, input_dim, kernel_size=3):
        """
        初始化CNN融合模块
        
        Args:
            input_dim: 输入矩阵的列数（特征维度）
            kernel_size: 卷积核大小，必须为奇数（默认为3）
        """
        super(CNNFusion, self).__init__()
        
        # 确保kernel_size为奇数
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        
        # 1D卷积层
        # in_channels: 输入特征数
        # out_channels: 输出特征数（与输入相同，保持维度）
        # kernel_size: 卷积核大小
        # stride: 步长为2，实现隔行采样
        # padding: 填充，保持边缘信息
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=kernel_size,
            stride=2,              # 步长2，每两行融合为一行
            padding=kernel_size // 2  # 填充以保持长度
        )
        
        # 激活函数
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入矩阵，形状为 (batch, seq_len, input_dim)
            
        Returns:
            融合后的矩阵，形状为 (batch, seq_len//2, input_dim)
        """
        # print(f"  CNNFusion Input Shape (before permute): {x.shape}")
        # print(f"  CNNFusion Input Values (before permute): {x.detach().numpy()}")
        # 调整维度: (batch, seq_len, input_dim) → (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        # print(f"  CNNFusion After Permute 1 Shape: {x.shape}")
        # print(f"  CNNFusion After Permute 1 Values: {x.detach().numpy()}")

        # 卷积操作
        x = self.conv(x)
        # print(f"  CNNFusion After Conv Shape: {x.shape}")
        # print(f"  CNNFusion After Conv Values: {x.detach().numpy()}")
        
        # 激活和归一化
        x = self.leaky_relu(x)
        # print(f"  CNNFusion After LeakyReLU Shape: {x.shape}")
        # print(f"  CNNFusion After LeakyReLU Values: {x.detach().numpy()}")
        
        # 调整维度: (batch, input_dim, seq_len) → (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1)
        # print(f"  CNNFusion After Permute 2 Shape: {x.shape}")
        # print(f"  CNNFusion After Permute 2 Values: {x.detach().numpy()}")

        x = self.layer_norm(x)
        # print(f"  CNNFusion After LayerNorm Shape: {x.shape}")
        # print(f"  CNNFusion After LayerNorm Values: {x.detach().numpy()}")
        
        return x

    def save_model(self, path):
        """保存模型参数"""
        torch.save(self.state_dict(), path)
        print(f"CNNFusion模型已保存到: {path}")

    @classmethod
    def load_model(cls, path, input_dim, kernel_size=3):
        """加载模型参数"""
        model = cls(input_dim, kernel_size)
        model.load_state_dict(torch.load(path))
        model.eval()
        print(f"CNNFusion模型已从 {path} 加载")
        return model


def cnn_fuse_rows(input_matrix, kernel_size=3, model_path=None):
    """
    使用CNN融合矩阵的相邻两行
    
    Args:
        input_matrix: 输入矩阵，numpy数组，形状为 (seq_len, input_dim)
        kernel_size: 卷积核大小
        model_path: 模型保存/加载路径，如果为None则不保存/加载
        
    Returns:
        融合后的矩阵，numpy数组，形状为 (seq_len//2, input_dim)
    """
    # 获取输入维度
    input_dim = input_matrix.shape[1]
    seq_len = input_matrix.shape[0]
    
    # 确保行数为偶数
    if seq_len % 2 != 0:
        raise ValueError(f"输入矩阵行数 ({seq_len}) 必须为偶数")
    
    # 创建或加载CNN融合模型
    if model_path is not None and os.path.exists(model_path):
        print(f"  加载已保存的CNNFusion模型: {model_path}")
        cnn_model = CNNFusion.load_model(model_path, input_dim, kernel_size)
    else:
        cnn_model = CNNFusion(
            input_dim=input_dim,
            kernel_size=kernel_size
        )
        if model_path is not None:
            print(f"  保存CNNFusion模型到: {model_path}")
            cnn_model.save_model(model_path)
    
    # 设置为评估模式
    cnn_model.eval()
    
    # 转换为张量
    # 添加batch维度: (seq_len, input_dim) → (1, seq_len, input_dim)
    input_tensor = torch.tensor(input_matrix, dtype=torch.float32).unsqueeze(0)
    
    # 执行融合
    with torch.no_grad():
        output_tensor = cnn_model(input_tensor)
    
    # 转换回numpy数组
    # 移除batch维度: (1, seq_len//2, input_dim) → (seq_len//2, input_dim)
    output_matrix = output_tensor.squeeze(0).numpy()
    
    return output_matrix


def demo_cnn_fusion():
    """
    演示CNN融合功能
    """
    # 获取当前脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "QualitativeMatrix.txt")
    
    print("=" * 60)
    print("步骤1: 加载原始矩阵数据")
    print("=" * 60)
    
    # 加载原始数据
    original_matrix = load_matrix_from_file(data_file)
    print(f"原始矩阵形状: {original_matrix.shape}")
    print("原始矩阵:")
    print(original_matrix)
    
    print("\n" + "=" * 60)
    print("步骤2: 多头注意力融合")
    print("=" * 60)
    
    # 先进行多头注意力融合
    attention_output = fuse_matrix_with_attention(
        original_matrix,
        num_heads=2,
        dropout=0.0
    )
    print(f"多头注意力输出形状: {attention_output.shape}")
    print("多头注意力输出:")
    # print(attention_output)
    
    print("\n" + "=" * 60)
    print("步骤3: CNN行融合（每两行融合为一行）")
    print("=" * 60)
    
    # 使用CNN进行行融合
    cnn_output = cnn_fuse_rows(attention_output, kernel_size=3)
    print(f"CNN融合输出形状: {cnn_output.shape}")
    print("CNN融合输出:")
    # print(cnn_output)
    
    print("\n" + "=" * 60)
    print("维度变化总结")
    print("=" * 60)
    print(f"原始矩阵: {original_matrix.shape}")
    print(f"多头注意力后: {attention_output.shape}")
    print(f"CNN融合后: {cnn_output.shape}")
    print(f"行数减半: {original_matrix.shape[0]} → {cnn_output.shape[0]} (减半)")
    print(f"列数保持: {original_matrix.shape[1]} → {cnn_output.shape[1]} (不变)")
    
    # 验证维度一致性
    expected_rows = original_matrix.shape[0] // 2
    print(f"\n维度验证:")
    print(f"期望行数: {expected_rows}, 实际行数: {cnn_output.shape[0]}")
    print(f"维度是否正确: {cnn_output.shape[0] == expected_rows}")


if __name__ == "__main__":
    demo_cnn_fusion()
