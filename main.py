import time
from argparse import ArgumentParser

import termcolor
from transformers import AutoTokenizer
from auto_gptq.modeling._const import SUPPORTED_MODELS

from gpt_bigcode_gptq import GPTBigCodeGPTQForCausalLM


if "gpt_bigcode" not in SUPPORTED_MODELS:
    SUPPORTED_MODELS.append("gpt_bigcode")


def load_tokenizer(pretrained_model_name):
    pretrained_model_name = "bigcode/starcoder"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    return tokenizer


def load_model(quantized_model_dir, args):
    quantized_model_dir = "starcoder-4bit-128g"
    model = GPTBigCodeGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", trust_remote_code=True)
    return model


def generate(model, tokenizer, args):
    batch = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=True)
    batch = {k: v.cuda() for k, v in batch.items()}

    for _ in range(2):
        print("generating...")
        t1 = time.time()
        generated = model.generate(batch["input_ids"], do_sample=False, min_new_tokens=100, max_new_tokens=100)
        t2 = time.time()
        print(termcolor.colored(tokenizer.decode(generated[0]), "yellow"))
        print("generated in %0.2fms" % ((t2 - t1) * 1000))

    print("prompt tokens", len(batch["input_ids"][0]))
    print("all tokens", len(generated[0]))

    generated_tokens = len(generated[0]) - len(batch["input_ids"][0])
    print("%0.1fms per token" % (((t2 - t1) * 1000) / generated_tokens))
    # print(tokenizer.decode(model.generate(**tokenizer(args.prompt, return_tensors="pt").to("cuda:0"), temperature=0.4, max_length=512)[0]))


def main():
    parser = ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="bigcode/starcoder", help="tokenizer to load, such as bigcode/starcoder")
    parser.add_argument("--model_dir", type=str, default="starcoder-4bit-128g", help="model directory to load, such as starcoder-4bit-128g")
    parser.add_argument("--load", type=str, help="load a quantized checkpoint, use normal model if not specified")
    parser.add_argument(
        "--groupsize", type=int, default=-1, help="Groupsize to use for quantization; default uses full row."
    )
    parser.add_argument("--prompt", type=str, help="prompt the model")
    args = parser.parse_args()

    t1 = time.time()
    tokenizer = load_tokenizer(args.tokenizer)
    model = load_model(args.model_dir, args)
    t2 = time.time()
    print("model and tokenizer load time %0.1fms" % ((t2 - t1) * 1000))

    
    t1 = time.time()
    generate(model, tokenizer, args)
    t2 = time.time()
    print("generation time %0.1fms" % ((t2 - t1) * 1000))


if __name__ == "__main__":
    main()