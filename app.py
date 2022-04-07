import numpy as np
from transformers import BertTokenizer
import tensorrt as trt
import common
import os
from sanic import Sanic, response
from sanic_cors import CORS
from sanic_openapi import swagger_blueprint, doc
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
engine_model_path = "bert_fp16.trt"



# Build a TensorRT engine.

# Contexts are used to perform inference.

def get_engine(engine_file_path):
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        return engine




"""
b、从engine中获取inputs, outputs, bindings, stream 的格式以及分配缓存
"""

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
app = Sanic(__name__)
CORS(app)
app.blueprint(swagger_blueprint)
app.config["API_VERSION"] = "0.1"
app.config["API_TITLE"] = "DIALOG_SERVICE：Sanci-OpenAPI"

server_port = int(os.getenv('SERVER_PORT', 8181))
@app.post("/bot/message")
@doc.summary("Let us have a chat")
@doc.consumes(doc.JsonBody({"message1": str,"message2":str}), location="body")
def message(request):
    res = []
    message1 = request.json.get("message1")
    sentence1 = message1
    sentence2 = request.json.get("message2")
    engine = get_engine(engine_model_path)
    context = engine.create_execution_context()


    inputs = tokenizer.encode_plus(sentence1,sentence2, return_tensors='pt', add_special_tokens=True)
    tokens_id =  to_numpy(inputs['input_ids'].int())
    attention_mask = to_numpy(inputs['attention_mask'].int())
    context.active_optimization_profile = 0
    origin_inputshape = context.get_binding_shape(0)                # (1,-1)
    origin_inputshape[0],origin_inputshape[1] = tokens_id.shape     # (batch_size, max_sequence_length)
    context.set_binding_shape(0, (origin_inputshape))
    context.set_binding_shape(1, (origin_inputshape))

    """
    c、输入数据填充
    """
    inputs, outputs, bindings, stream = common.allocate_buffers_v2(engine, context)
    inputs[0].host = tokens_id
    inputs[1].host = attention_mask

    """
    d、tensorrt推理
    """
    trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    preds = np.argmax(trt_outputs, axis=1)
    print("====preds====:",preds)
    labels = ["agreed","disagreed","unrelated"]
    res.append(labels[int(preds[0])])
    return request.json(res)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=server_port)