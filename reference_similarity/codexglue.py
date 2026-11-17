import os
import json
import torch
import openpyxl
import sys
from transformers import AutoTokenizer, AutoModel

# 目录迁移及模型配置兼容：按现有CodeBERT配置选择加载类，评分函数沿用原代码。
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "outputs", "models", "codebert-base")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True)

# 计算代码相似度
def get_code_similarity(code1: str, code2: str) -> float:
    def encode(code):
        inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            vec = model(**inputs).last_hidden_state[:, 0, :]
        return vec
    v1, v2 = encode(code1), encode(code2)
    return round(torch.cosine_similarity(v1, v2).item(), 4)

# 处理代码内容
def process_code(code_content):
    if not code_content:
        return ""
    code = code_content.replace('\\n', '\n').strip()
    if code.startswith('```python'):
        code = code[9:]
    elif code.startswith('```'):
        code = code[3:]
    code = code.strip()
    if code.endswith('```'):
        code = code[:-3]
    code = code.strip()
    return code

# 评价Excel文件中所有行的代码相似度，并将结果写入Excel文件
def evaluate_codeXGLUE(excel_file, source_column='canonical_solution', target_column='answer'):
    """
    评估Excel文件中指定列代码的相似度，并将结果写入Excel文件
    
    Args:
        excel_file: Excel文件路径
        source_column: 参考代码列名（默认: canonical_solution）
        target_column: 生成代码列名（默认: answer）
    """
    try:
        # 读取Excel文件
        print("读取Excel文件...")
        wb = openpyxl.load_workbook(excel_file)
        ws = wb.active
        
        # 获取列名
        headers = [cell.value for cell in ws[1]]
        
        # 检查是否存在指定的列
        if source_column not in headers:
            print(f"错误: Excel文件中未找到{source_column}列")
            return
        if target_column not in headers:
            print(f"错误: Excel文件中未找到{target_column}列")
            return
        
        # 获取列索引
        completion_index = headers.index(source_column)
        answer_index = headers.index(target_column)
        
        # 检查是否存在CodeXGLUE列，如果不存在则添加
        result_column = 'CodeXGLUE'
        if result_column not in headers:
            headers.append(result_column)
            # 使用正确的方式添加新列
            ws.cell(row=1, column=len(headers), value=result_column)
        
        # 获取结果列的索引
        codexglue_index = headers.index(result_column)
        
        # 遍历所有数据行（从第2行开始）
        total_rows = ws.max_row
        print(f"开始处理Excel文件，共{total_rows-1}行数据...")
        
        for row_num in range(2, total_rows + 1):
            try:
                # 获取当前行数据
                completion_value = ws[row_num][completion_index].value
                answer_value = ws[row_num][answer_index].value
                
                if not completion_value or not answer_value:
                    print(f"第{row_num-1}条数据的completion或answer列为空，跳过处理")
                    ws[row_num][codexglue_index].value = None
                    continue
                
                # 打印处理进度
                if (row_num-1) % 10 == 0:
                    print(f"处理第{row_num-1}条数据...")
                
                # 处理代码
                completion_code = process_code(completion_value)
                answer_code = process_code(answer_value)
                
                if not completion_code or not answer_code:
                    print(f"第{row_num-1}条数据的completion或answer列代码为空，跳过处理")
                    ws[row_num][codexglue_index].value = None
                    continue
                
                # 计算相似度
                score = get_code_similarity(completion_code, answer_code)
                
                # 写入结果到Excel文件
                ws[row_num][codexglue_index].value = score
                
            except Exception as e:
                print(f"处理第{row_num-1}条数据时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                ws[row_num][codexglue_index].value = None
                continue
        
        # 保存Excel文件
        output_file = os.path.splitext(excel_file)[0] + "_with_codexglue.xlsx"
        wb.save(output_file)
        print(f"\n处理完成！结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 示例：直接调用 evaluate_codeXGLUE 函数
    excel_file = os.path.join(os.path.dirname(SCRIPT_DIR), 'outputs', 'runs', 'working_with_levenshtein.xlsx')
    source_column = "canonical_solution"
    target_column = "answer"
    evaluate_codeXGLUE(excel_file, source_column, target_column)
