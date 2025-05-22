# -*- coding: utf-8 -*-
"""
Automatically generated 
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig

lora_path = #Load the model from the Github
peft_config = PeftConfig.from_pretrained(lora_path)

base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    device_map="auto",
    torch_dtype="auto"
)

model = PeftModel.from_pretrained(base_model, lora_path)
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

from transformers import pipeline

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map = "auto")

pipe("Instruction: Who are you? \nInput: \nOutput:", max_new_tokens=300, do_sample=True, temperature=0.9, top_p=0.95)

