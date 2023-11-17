from transformers import pipeline
import torch

gen = pipeline('text-generation', model ='EleutherAI/gpt-neo-2.7B')
context = "Deep Learning is a sub-field of Artificial Intelligence."
output = torch.generator(context, max_length=50, do_sample=True, temperature=0.9)

with open('dl.txt', 'w') as f:
    f.write(str(output))

print(output)