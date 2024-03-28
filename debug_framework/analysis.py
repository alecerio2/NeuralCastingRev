import onnx
import os
import onnxruntime
import numpy as np

def inputs_reimplemented_gru_1():
    inputs = {
        'onnx::Gemm_0': np.ones((1, 3), dtype=np.float32),
        'onnx::Gemm_1': np.zeros((1, 4), dtype=np.float32)
    }
    return inputs

def inputs_nsnet2():
    inputs = {
        'in_noisy': np.ones((1, 1, 257), dtype=np.float32),
        'h1': np.zeros((1, 1, 400), dtype=np.float32),
        'h2': np.zeros((1, 1, 400), dtype=np.float32)
    }
    return inputs

##############################################################################
#                       UPDATE INPUTS MODEL
##############################################################################

inputs = inputs_nsnet2()

##############################################################################
##############################################################################

script_dir = os.path.dirname(os.path.realpath(__file__))

onnx_file_names = []
file_names = os.listdir(script_dir + '/output/')
for file_name in file_names:
    base_name, extension = os.path.splitext(file_name)
    if extension == '.onnx':
        onnx_file_names.append(file_name)

output_txt = ""
for onnx_name in onnx_file_names:

    model_path = script_dir + '/output/' + onnx_name
    model = onnx.load(model_path)

    # Get the input nodes
    input_nodes = [input.name for input in model.graph.input]

    session = onnxruntime.InferenceSession(model_path)

    curr_inputs_inference = {}
    curr_inputs = session.get_inputs()
    for input in curr_inputs:
        val = inputs[input.name]
        curr_inputs_inference[input.name] = val


    output = session.run([], curr_inputs_inference)
    
    output_txt += "##############################################################################\n"
    output_txt += "Unit name: " + onnx_name + "\n"
    output_txt += "Model output:\n"
    output_txt += str(output) + "\n\n"
    output_txt += "Model output shape:\n"
    output_txt += str(output[0].shape) + "\n\n"

with open(script_dir + '/output.txt', 'w') as file:
    file.write(output_txt)