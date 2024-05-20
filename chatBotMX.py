# from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, pipeline

import numpy as np
from flask import Flask, request, jsonify
import transformers
import torch


model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"


pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

pipeline.model.eval()

PROMPT = '''당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다.
You are a helpful AI assistant, you'll need to answer users' queries in a friendly and accurate manner.'''
instruction = "서울과학기술대학교 MLP연구실에 대해 소개해줘"

messages = [
    {"role": "system", "content": f"{PROMPT}"},
    {"role": "user", "content": f"{instruction}"}
    ]

prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=2048,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9
)

print(outputs[0]["generated_text"][len(prompt):])



app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.get_json()['i love coding for sure']
    inputs = pipeline.tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=512,
        return_attention_mask=True,
        return_tensors='pt'
    )

    outputs = pipeline.model(**inputs)
    logits = outputs.logits
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

    return jsonify({'probs': probs.tolist()})


if __name__ == "__main__":
  # Flask 서버 실행
  app.run(host="0.0.0.0", port=5000)




