from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from embed_models import ColOCR
import torch


MODEL_NAME = "deepseek-ai/DeepSeek-OCR"

# tokenizer (same vocab/special tokens as the base model)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 3) Load pretrained weights *directly* into your subclass
colocr = ColOCR.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    _attn_implementation="flash_attention_2",  # or "eager" if FA2 complains on your stack
    # device_map="auto",  # or .to("cuda") after load
).to("cuda").eval()

# sanity checks
print(type(colocr))                          # <class '__main__.ColOCR'>
# breakpoint()
print(next(colocr.parameters()).dtype)       # torch.bfloat16
print(colocr.__class__.__mro__[:4])          # shows your subclass and Deepseek classes in the MRO

# tiny smoke test (text-only)
# inputs = tokenizer("Hello", return_tensors="pt").to(colocr.device)
# with torch.no_grad(), torch.autocast(colocr.device.type, dtype=torch.bfloat16):
#     out = colocr.generate(**inputs, max_new_tokens=10, do_sample=False)
# print(tokenizer.decode(out[0], skip_special_tokens=True))

prompt = "<image>\n<|grounding|>Convert the document to markdown. "
# -> vit encoder
# prompt = "300 image tokens \n<|grounding|>200 text tokens. "; query = "100 text tokens"
# -> LLM decoder
# the recovered document.
# "prompt + original_text_tokens"<1st><23456783456784567><2nd>
# p(\cdot|prompt)

# image_file = 'pages/page_001.png'
image_file = './pages/one_table.png'
# image_file = ['pages/page_001.png', 'pages/page_002.png']
output_path = './'

# infer(self, tokenizer, prompt='', image_file='', output_path = ' ', base_size = 1024, image_size = 640, crop_mode = True, test_compress = False, save_results = False):

# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False

# Gundam: base_size = 1024, image_size = 640, crop_mode = True

res = colocr.infer(tokenizer, prompt=prompt, image_file=image_file, output_path=output_path, base_size=640,
                  image_size=640, crop_mode=False, save_results=True, test_compress=True)
