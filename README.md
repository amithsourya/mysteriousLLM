# Mysterious LLM
A compact yet powerful Large Language Model (LLM) built on top of Microsoft’s Phi architecture, optimized for giving mysterious responses within a small compute footprint. This model represents a fine-tuned variant of the Phi series with upto 2.7 Billion Parameters

- **Developed by:** Amith Sourya Sadineni
- **Model type:** Text Generation
- **Language(s):** Python
- **License:** MIT
- **Finetuned from model:** microsoft/phi-2

### Model Sources

<!-- Provide the basic links for the model. -->
- **Repo:**: https://github.com/amithsourya/mysteriousLLM/blob/main/content/entity_lora_adapter/adapter_model.safetensors
- **Demo:**
```python
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
```

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** T4 GPU

## Example
### After fine tuning
![image](https://github.com/user-attachments/assets/dd77d6dd-9f5a-46a2-905c-f6c41c10fc6c)


