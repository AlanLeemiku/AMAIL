import yaml

# 加载配置文件
config_file = "configs/push/1.00/drail.yaml"  # 请根据实际路径调整

with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

# 提取参数并整理成命令行参数格式
def format_parameters(parameters):
    formatted_params = []
    for param, value in parameters.items():
        if isinstance(value, dict) and 'values' in value:  # 如果是有多种取值的参数
            formatted_params.append(f"-{param} {value['values']}")
        else:
            formatted_params.append(f"-{param} {value['value']}")
    return formatted_params

# 获取参数
parameters = config.get('parameters', {})

# 格式化参数
formatted_parameters = format_parameters(parameters)

# 打印整理后的参数
print(" ".join(formatted_parameters))
