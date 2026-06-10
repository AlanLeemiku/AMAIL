import wandb
import subprocess
import yaml
import os

# 加载配置文件（这里假设配置文件路径是 config.yaml）
config_file = "./configs/push/1.00/drail.yaml"  
# 读取配置文件
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

# 启动 wandb 实验并记录相关信息
wandb.init(
    project=config['project'],                # 项目名称
    config=config['parameters'],              # 配置参数
    name=config['name'],                      # 实验名称                # 搜索方法（例如: grid）
)
config_file = "configs/push/1.00/drail.yaml"  # 请根据实际路径调整
# 提取参数并整理成命令行参数格式
def format_parameters(parameters):
    formatted_params = []
    for param, value in parameters.items():
        if isinstance(value, dict) and 'values' in value:  # 如果是有多种取值的参数
            formatted_params.append(f"--{param} {value['values']}")
        else:
            formatted_params.append(f"--{param} {value['value']}")
    return formatted_params
# 获取参数
parameters = config.get('parameters', {})
# 格式化参数
program_args = format_parameters(parameters)
# 设置环境变量和程序参数
program = config['program']
# 构建程序命令
cmd = ["python3", program] + program_args +["--use-proper-time-limits"]
print(f"Running command: {' '.join(cmd)}")
# 运行程序（可以根据需要调整或传递额外的环境变量）
subprocess.run(cmd, check=True)
# 结束 wandb 监控
wandb.finish()
