import transformers
import torch
import gc

# transformers의 로깅 레벨을 경고 이상으로 설정
transformers.logging.set_verbosity_error()

model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"

# 모델을 초기화
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# 모델을 평가 모드로 설정
pipeline.model.eval()

def generate_text(instruction):
    try:
        # Define the prompt and messages
        PROMPT = '''당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다.
        You are a helpful AI assistant, you'll need to answer users' queries in a friendly and accurate manner.'''
        
        messages = [
            {"role": "system", "content": f"{PROMPT}"},
            {"role": "user", "content": f"{instruction}"}
        ]

        # Create the prompt for the model
        prompt = pipeline.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Ensure eos_token_id is valid
        eos_token_id = pipeline.tokenizer.eos_token_id
        empty_token_id = pipeline.tokenizer.convert_tokens_to_ids("")

        # Handle potential NoneType for empty_token_id
        if empty_token_id is None:
            empty_token_id = eos_token_id

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("")
        ]

        # Generate the text
        outputs = pipeline(
            prompt,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

        # Get the generated text
        generated_text = outputs[0]["generated_text"][len(prompt):]
        return generated_text

    finally:
        # Clean up to free GPU memory
        torch.cuda.empty_cache()
        gc.collect()
