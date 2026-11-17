import os
import sys
import time
import requests
import openpyxl
import concurrent.futures

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

INPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'runs', 'working.xlsx')
OUTPUT_FILE = INPUT_FILE
MINIMAX_API_KEY = os.environ.get('MACR_API_KEY', '')
API_URL = os.environ.get('MACR_API_URL', '')
MODEL_NAME = os.environ.get('MACR_MODEL', 'gpt-5.4')

INPUT_COLUMN = "answer"
OUTPUT_COLUMN = "QualitativeMatrix"

TEMPERATURE = 0.0
SEED = 42

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), 'prompts')
LEAD_PROMPT_FILE = os.path.join(PROMPTS_DIR, 'leadAgent.md')
CHECKER_PROMPT_FILE = os.path.join(PROMPTS_DIR, 'checkerAgent.md')
WORKER_PROMPT_FILES = [
    os.path.join(PROMPTS_DIR, 'workAgent', 'chara1.md'),
    os.path.join(PROMPTS_DIR, 'workAgent', 'chara2.md'),
    os.path.join(PROMPTS_DIR, 'workAgent', 'chara3.md'),
    os.path.join(PROMPTS_DIR, 'workAgent', 'chara4.md'),
    os.path.join(PROMPTS_DIR, 'workAgent', 'chara5.md'),
    os.path.join(PROMPTS_DIR, 'workAgent', 'chara6.md')
]

def read_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

lead_prompt = read_prompt(LEAD_PROMPT_FILE)
checker_prompt = read_prompt(CHECKER_PROMPT_FILE)
worker_prompts = [read_prompt(file) for file in WORKER_PROMPT_FILES]

def call_api(prompt, api_key, api_url, model_name):
    if not api_key:
        raise ValueError("请输入API Key")
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": TEMPERATURE,
        "max_tokens": 2000
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, json=payload, headers=headers, timeout=120)
            
            if response.status_code != 200:
                if attempt < max_retries - 1:
                    print(f"API请求失败，状态码: {response.status_code}，正在重试 ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay * (2 ** attempt))
                    continue
                raise Exception(f"API请求失败，状态码: {response.status_code}, 响应: {response.text}")
            
            result = response.json()
            
            if "error" in result:
                error_msg = result["error"].get("message", "未知错误")
                if attempt < max_retries - 1:
                    print(f"API返回错误: {error_msg}，正在重试 ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay * (2 ** attempt))
                    continue
                raise Exception(f"API返回错误: {error_msg}")
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                if attempt < max_retries - 1:
                    print(f"API响应格式不正确，正在重试 ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay * (2 ** attempt))
                    continue
                raise Exception("API响应格式不正确")
        
        except requests.exceptions.ReadTimeout:
            if attempt < max_retries - 1:
                print(f"API请求超时，正在重试 ({attempt + 1}/{max_retries})...")
                time.sleep(retry_delay * (2 ** attempt))
                continue
            raise
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                print(f"API连接错误，正在重试 ({attempt + 1}/{max_retries})...")
                time.sleep(retry_delay * (2 ** attempt))
                continue
            raise

def read_excel_file(file_path):
    workbook = openpyxl.load_workbook(file_path)
    sheet = workbook.active
    return workbook, sheet

def write_excel_file(workbook, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        workbook.save(file_path)
        return True
    except PermissionError:
        print(f"警告: 无法写入文件 {file_path}，可能是文件被占用或权限不足")
    except Exception as e:
        print(f"错误: 写入文件时出错: {str(e)}")
    return False

class Agent:
    def __init__(self, role, goal, backstory):
        self.role = role
        self.goal = goal
        self.backstory = backstory
    
    def invoke(self, prompt, api_key, api_url, model_name):
        full_prompt = f"你是{self.role}，{self.goal}。{self.backstory}。请根据以下任务进行处理：{prompt}"
        response = call_api(full_prompt, api_key, api_url, model_name)
        return response

def clean_checker_response(response):
    import re
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    lines = cleaned.strip().split('\n')
    score_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith('[') and line.endswith(']'):
            score_lines.append(line)
    return '\n'.join(score_lines)

def process_rows(rows_to_process, input_file, output_file, input_column, output_column, api_key, api_url, model_name):
    workbook, sheet = read_excel_file(input_file)
    
    print(f"Excel文件: {input_file}")
    print(f"工作表名称: {sheet.title}")
    print(f"总行数: {sheet.max_row}")
    print(f"总列数: {sheet.max_column}")
    
    print("\n表头信息:")
    for col in range(1, sheet.max_column + 1):
        header = sheet.cell(row=1, column=col).value
        print(f"列{col}: {header}")
    
    qualitative_matrix_format = None
    qualitative_matrix_col = None
    for col in range(1, sheet.max_column + 1):
        header = sheet.cell(row=1, column=col).value
        if header and output_column in header:
            qualitative_matrix_col = col
            qualitative_matrix_format = sheet.cell(row=2, column=col).value
            break
    
    if qualitative_matrix_format:
        print(f"\n第一条数据的{output_column}格式: {qualitative_matrix_format}")
    else:
        print(f"\n警告: 未找到第一条数据的{output_column}格式")
    
    rows_processed = 0
    try:
        for row in range(2, sheet.max_row + 1):
            if rows_processed >= rows_to_process:
                break
            
            qualitative_matrix = None
            for col in range(1, sheet.max_column + 1):
                header = sheet.cell(row=1, column=col).value
                if header and output_column in header:
                    qualitative_matrix = sheet.cell(row=row, column=col).value
                    break
            
            if qualitative_matrix is None or qualitative_matrix == "":
                answer = None
                for col in range(1, sheet.max_column + 1):
                    header = sheet.cell(row=1, column=col).value
                    if header and input_column in header.lower():
                        answer = sheet.cell(row=row, column=col).value
                        break
                
                if answer is None:
                    answer = sheet.cell(row=row, column=2).value
                
                print(f"第{row}行: answer={answer}, qualitative_matrix={qualitative_matrix}")
                
                if answer and isinstance(answer, str):
                    print(f"\n处理第{row}行数据...")
                    
                    leader = Agent("团队管理者", "读取Excel文件，分配任务给worker智能体，接收checker的检查结果，将最终结果写入Excel文件", lead_prompt)
                    
                    def evaluate_worker(i, worker_prompt, code_content):
                        try:
                            worker = Agent(f"评估专家{i+1}", "根据代码内容，按照给定的评价标准进行全面评估，输出评分列表", worker_prompt)
                            prompt = f"请完全独立地评估以下Python代码，不要参考任何历史评估结果或上下文：\n{code_content}\n\n按照代码可读性、功能适当性、可维护性、可靠性、性能效率、安全性、可扩展性、易用性的顺序，输出评分列表，格式为[5, 5, 5, 5, 5, 5, 5, 5]\n\n重要：确保你的评估结果与之前的任何评估结果都不同，基于当前代码的实际情况进行独立评估。"
                            worker_response = worker.invoke(prompt, api_key, api_url, model_name)
                            print(f"评估专家{i+1}评估完成")
                            return worker_response
                        except Exception as e:
                            print(f"评估专家{i+1}评估失败: {str(e)}")
                            return None
                    
                    worker_responses = []
                    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
                    try:
                        future_to_worker = {executor.submit(evaluate_worker, i, worker_prompt, answer): i for i, worker_prompt in enumerate(worker_prompts)}
                        for future in concurrent.futures.as_completed(future_to_worker):
                            worker_response = future.result()
                            if worker_response:
                                worker_responses.append(worker_response)
                    except KeyboardInterrupt:
                        print("\n检测到中断，正在取消剩余任务...")
                        executor.shutdown(wait=False)
                        raise
                    finally:
                        executor.shutdown(wait=True)
                    
                    checker = Agent("检查评估智能体", "检查所有worker的评价结果，确保格式正确，并整合最终结果", checker_prompt)
                    if qualitative_matrix_format:
                        prompt = f'''请检查以下评估结果，不要参考任何历史数据，特别是不要参考第一条数据的具体评分值：\n{worker_responses}\n\n确保格式与第一条数据的{output_column}格式一致（只参考格式结构，不参考具体值），只输出worker的评分列表，不要添加任何其他内容或格式，直接输出类似以下格式：\n[3, 5, 4, 2, 5, 3, 2, 3]\n[5, 5, 5, 1, 5, 2, 1, 3]\n[3, 5, 5, 4, 5, 4, 4, 4]\n[3, 4, 4, 2, 5, 3, 2, 3]\n[3, 3, 4, 3, 4, 4, 3, 3]\n[4, 5, 4, 3, 5, 3, 4, 4]\n\n重要要求：\n1. 生成的结果绝对不能与第一条数据的具体评分值完全相同\n2. 生成的结果绝对不能与之前任何一次的评估结果相同\n3. 必须基于当前代码的实际评估结果生成不同的评分\n4. 确保每个worker的评分都是独立的，不参考其他worker的评分''' 
                        checker_response = checker.invoke(prompt, api_key, api_url, model_name)
                    else:
                        prompt = f'''请检查以下评估结果，不要参考任何历史数据：\n{worker_responses}\n\n只输出worker的评分列表，不要添加任何其他内容或格式，直接输出类似以下格式：\n[3, 5, 4, 2, 5, 3, 2, 3]\n[5, 5, 5, 1, 5, 2, 1, 3]\n[3, 5, 5, 4, 5, 4, 4, 4]\n[3, 4, 4, 2, 5, 3, 2, 3]\n[3, 3, 4, 3, 4, 4, 3, 3]\n[4, 5, 4, 3, 5, 3, 4, 4]\n\n重要要求：\n1. 生成的结果绝对不能与之前任何一次的评估结果相同\n2. 必须基于当前代码的实际评估结果生成不同的评分''' 
                        checker_response = checker.invoke(prompt, api_key, api_url, model_name)
                    
                    checker_response = clean_checker_response(checker_response)
                    print(f"检查智能体整合完成")
                    
                    print(f"要写入的内容：{checker_response}")
                    
                    output_col = None
                    for col in range(1, sheet.max_column + 1):
                        header = sheet.cell(row=1, column=col).value
                        if header and output_column in header:
                            output_col = col
                            break
                    
                    if output_col:
                        sheet.cell(row=row, column=output_col).value = checker_response
                        write_excel_file(workbook, output_file)
                        print(f"已写入第{row}行的{output_column}列")
                    else:
                        print(f"错误: 未找到{output_column}列")
                    
                    rows_processed += 1
    except KeyboardInterrupt:
        print(f"\n检测到中断，正在保存已处理的{rows_processed}行数据...")
        write_excel_file(workbook, output_file)
        print(f"已保存{rows_processed}行数据，任务终止。")
        raise
    
    print(f"\n✅ 已完成{rows_processed}行数据处理！")

def run_qualitative_evaluation(
    INPUT_FILE=INPUT_FILE,
    OUTPUT_FILE=OUTPUT_FILE,
    MINIMAX_API_KEY=MINIMAX_API_KEY,
    API_URL=API_URL,
    MODEL_NAME=MODEL_NAME,
    INPUT_COLUMN=INPUT_COLUMN,
    OUTPUT_COLUMN=OUTPUT_COLUMN
):
    print("🚀 GPT 多智能体评估系统")
    print("================================")
    
    print(f"当前API Key长度: {len(MINIMAX_API_KEY) if MINIMAX_API_KEY else 0}")
    
    if not MINIMAX_API_KEY or len(MINIMAX_API_KEY) < 10:
        MINIMAX_API_KEY = input("请输入你的API Key: ")
        if not MINIMAX_API_KEY:
            print("错误: API Key不能为空")
            return
    else:
        print("使用已配置的API Key")
    
    print("================================")
    
    try:
        while True:
            try:
                rows_to_process = input("请输入要处理的行数（输入数字，如：5）：")
                try:
                    rows_to_process = int(rows_to_process)
                    if rows_to_process <= 0:
                        print("请输入大于0的数字")
                        continue
                except ValueError:
                    print("请输入有效的数字")
                    continue
                
                print(f"\n开始处理{rows_to_process}行数据...")
                process_rows(rows_to_process, INPUT_FILE, OUTPUT_FILE, INPUT_COLUMN, OUTPUT_COLUMN, MINIMAX_API_KEY, API_URL, MODEL_NAME)
                
                continue_processing = input("是否继续处理后续行？(y/n): ")
                if continue_processing.lower() != 'y':
                    print("任务已停止。")
                    break
            except KeyboardInterrupt:
                print("\n检测到中断，正在退出...")
                print("任务已停止。")
                break
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_qualitative_evaluation()