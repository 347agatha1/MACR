import os
import sys
import requests
import pandas as pd

# 设置编码环境变量
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# ===================== 配置 =====================
# 默认输入输出文件路径
INPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'runs', 'working.xlsx')
OUTPUT_FILE = INPUT_FILE

# 大模型配置 - 使用GPT
MINIMAX_API_KEY = os.environ.get('MACR_API_KEY', '')
API_URL = os.environ.get('MACR_API_URL', '')
MODEL_NAME = os.environ.get('MACR_MODEL', 'gpt-5.4')

# 默认列名
PROMPT_COLUMN = "prompt"
OUTPUT_COLUMN = "answer"
prompt_setChara_option = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.  "
is_use_setChara_option = True
# 可复现性配置
TEMPERATURE = 0.0  # 设置为0保证结果确定性（0=完全确定，1=最大随机性）
SEED = 42          # 固定随机种子，保证相同输入产生相同输出

# 对话历史配置（固定为False，每次只发送当前消息，不保存历史会话）
USE_CHAT_HISTORY = False  # True=保留对话历史，False=每次只发送当前消息

# ===================== API调用功能 =====================
def generate_code(prompt, api_key, api_url, model_name):
    """调用GPT API生成代码"""
    if not api_key:
        raise ValueError("请在代码中填入GPT API Key")
    
    if USE_CHAT_HISTORY:
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = [{"role": "user", "content": prompt}]
    
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "temperature": TEMPERATURE,
        # "seed": SEED,
        "max_tokens": 2000
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    print("发送请求到API...")
    response = requests.post(api_url, json=payload, headers=headers, timeout=60)
    
    if response.status_code != 200:
        raise Exception(f"API请求失败，状态码: {response.status_code}, 响应: {response.text}")
    
    result = response.json()
    
    if "error" in result:
        error_msg = result["error"].get("message", "未知错误")
        raise Exception(f"API返回错误: {error_msg}")
    
    if "choices" in result and len(result["choices"]) > 0:
        return result["choices"][0]["message"]["content"]
    else:
        raise Exception("API响应格式不正确")

# ===================== 主函数 =====================
def run_code_generator(
    input_file: str = INPUT_FILE,
    output_file: str = OUTPUT_FILE,
    api_key: str = MINIMAX_API_KEY,
    api_url: str = API_URL,
    model_name: str = MODEL_NAME,
    prompt_column: str = PROMPT_COLUMN,
    output_column: str = OUTPUT_COLUMN,
    start_index: int = 0,
    num_rows: int = 1
):
    """
    批量生成代码并写入Excel的指定列
    
    Args:
        input_file (str): 输入Excel文件路径
        output_file (str): 输出Excel文件路径
        api_key (str): API密钥
        api_url (str): API接口地址
        model_name (str): 模型名称
        prompt_column (str): 提示词所在列名
        output_column (str): 输出代码所在列名
        start_index (int): 数据起始行索引（默认为0，即第一条数据）
        num_rows (int): 要处理的行数（默认为1）
    """
    print("🚀 AI代码生成器")
    print("================================")
    print(f"起始索引: {start_index}")
    print(f"处理行数: {num_rows}")
    print(f"处理范围: {start_index} 到 {start_index + num_rows - 1}")
    print("================================")
    
    print(f"读取数据: {input_file}")
    
    try:
        df = pd.read_excel(input_file)
        print(f"数据读取成功，共 {len(df)} 条记录")
        
        if start_index < 0 or start_index >= len(df):
            raise Exception(f"起始索引 {start_index} 无效，有效范围: 0 到 {len(df)-1}")
        
        if num_rows <= 0:
            raise Exception(f"处理行数 {num_rows} 无效，必须大于0")
        
        end_index = start_index + num_rows - 1
        if end_index >= len(df):
            end_index = len(df) - 1
            print(f"警告: 结束索引超出范围，调整为 {end_index}")
        
        if prompt_column not in df.columns:
            raise Exception(f"数据中没有 '{prompt_column}' 列")
        
        if output_column not in df.columns:
            raise Exception(f"数据中没有 '{output_column}' 列")
        
        success_count = 0
        for idx in range(start_index, end_index + 1):
            print(f"\n{'='*60}")
            print(f"处理第 {idx} 行数据 (进度: {idx - start_index + 1}/{num_rows})")
            print(f"{'='*60}")
            
            try:
                query = df.iloc[idx][prompt_column]
                task_id = df.iloc[idx]['task_id']
                
                print(f"task_id: {task_id}")
                print(f"提示词长度: {len(str(query))} 字符")
                
                print("正在生成代码...")
                
                if is_use_setChara_option:
                    full_prompt = prompt_setChara_option + str(query)
                else:
                    full_prompt = str(query)
                
                code = generate_code(full_prompt, api_key, api_url, model_name)
                
                print(f"生成成功，代码长度: {len(code)} 字符")
                print(f"代码预览: {code[:200]}...")
                
                df.at[idx, output_column] = code
                print(f"代码已写入到第 {idx} 行的{output_column}列")
                
                success_count += 1
                
                df.to_excel(output_file, index=False)
                
            except Exception as e:
                print(f"第 {idx} 行处理失败: {str(e)}")
                continue
        
        print(f"\n{'='*60}")
        print("处理完成")
        print(f"{'='*60}")
        print(f"成功处理: {success_count}/{num_rows} 行")
        print(f"Excel文件已保存: {output_file}")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    start_index = 0
    num_rows = 2
    run_code_generator(start_index=start_index, num_rows=num_rows)
