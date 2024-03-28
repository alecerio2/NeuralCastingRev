import onnx
import numpy as np
from neural_cast.frontend.common.common import CompilerConfig
from onnxdbg.onnxdbg import onnxdbg
import os
import yaml

def data_gru_reimplemented_1(test_path, onnx_name):
    model = onnx.load(test_path + onnx_name + '.onnx')
    input_names = [input.name for input in model.graph.input]
    x_data =  np.ones((1, 3), dtype=np.float32)
    hidden_data =  np.zeros((1, 4), dtype=np.float32)
    input_data = {
        input_names[0]: x_data,
        input_names[1]: hidden_data
    }
    return input_data

def data_nsnet2(test_path, onnx_name):
    model = onnx.load(test_path + onnx_name + '.onnx')
    input_names = [input.name for input in model.graph.input]
    x_data =  np.ones((1, 1, 257), dtype=np.float32)
    h1 = np.zeros((1, 1, 400), dtype=np.float32)
    h2 = np.zeros((1, 1, 400), dtype=np.float32)
    input_data = {
        input_names[0]: x_data,
        input_names[1]: h1,
        input_names[2]: h2,
    }
    return input_data

def empty_output_folder(folder_path):
    file_names = os.listdir(folder_path)
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error occurred while removing file: {file_name}, {e}")

curr_file = os.path.abspath(__file__)
curr_path = os.path.dirname(curr_file)
with open(curr_path + '/../config/config.yaml', 'r') as yaml_file:
    config = yaml.safe_load(yaml_file)
CompilerConfig(config)

# init config file
test_path : str = CompilerConfig()['repo'] + 'debug_framework/'
output_path : str = test_path + 'output/'

##############################################################################
#                       UPDATE INPUTS MODEL
##############################################################################

# input data
onnx_name : str = 'nsnet2'
input_data = data_nsnet2(test_path, onnx_name)

##############################################################################
##############################################################################

empty_output_folder(output_path)

# run command inferdbg
onnxdbg("inferdbg", srcp=test_path, dstp=output_path, mdl=onnx_name, input=input_data)