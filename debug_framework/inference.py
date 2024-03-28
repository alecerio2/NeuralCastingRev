from neural_cast.compiler import run
from neural_cast.frontend.common.common import CompilerConfig
import os
import yaml

curr_file = os.path.abspath(__file__)
curr_path = os.path.dirname(curr_file)
with open(curr_path + '/../config/config.yaml', 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)
CompilerConfig(config)

# init config file
name : str = CompilerConfig()['name']
output_path : str = CompilerConfig()['output_path']
test_path : str = CompilerConfig()['repo'] + 'debug_framework/'
temp_path : str = CompilerConfig()['temp_path']
path_onnx = test_path + 'nsnet2.onnx'

# run compiler
run(CompilerConfig(), framework='onnx', path=path_onnx)

# create test main.c
#create_main_c(test_path, output_path, name)