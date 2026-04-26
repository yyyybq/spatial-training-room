# test_vllm.py
from vllm import LLM, SamplingParams

# 初始化（使用小模型测试）
llm = LLM(model="facebook/opt-125m",tensor_parallel_size=4)

# 生成参数
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# 测试
outputs = llm.generate(["Hello, my name is"], sampling_params)
for output in outputs:
    print(output.outputs[0].text)