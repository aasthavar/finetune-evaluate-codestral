import os, gc, torch, random
from dataclasses import dataclass, field
from datasets import load_dataset, load_from_disk
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
import bitsandbytes as bnb
from trl import SFTTrainer
from trl.commands.cli_utils import  TrlParser
from peft import LoraConfig, AutoPeftModelForCausalLM

os.environ["ACCELERATE_USE_FSDP"] = "1"
os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "1"

@dataclass
class ScriptArguments:
    ### training related
    dataset_path: str = field(default="/home/ubuntu/finetune-llms-on-aws/practise-fsdp/sft_cache/data")
    sm_save_model_dir: str = field(default="/opt/ml/model")
    
    model_id: str = field(default="openaccess-ai-collective/tiny-mistral")
    max_seq_length: int = field(default=4096)
    packing: bool = field(default=False)
    finetune_with_sm: bool = field(default=False)
    merge_weights_and_save: bool = field(default=False)
    save_tokenizer: bool = field(default=True)
    attn_implementation: str = field(default="sdpa")
    
    ### qlora related
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.1)
    task_type: str = field(default="CAUSAL_LM")
    
    ### bitsandbytes related
    load_in_4bit: bool = field(default=True)
    bnb_4bit_use_double_quant: bool = field(default=True)
    bnb_4bit_quant_type: str = field(default="nf4")
    bnb_4bit_compute_dtype: str = field(default="bfloat16")
    bnb_4bit_quant_storage: str = field(default="bfloat16")
    
def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)
    
def sft(script_args, training_args):
     # load dataset
    if script_args.finetune_with_sm:
        train_dataset = load_from_disk(os.path.join(script_args.dataset_path, "train"))
        test_dataset = load_from_disk(os.path.join(script_args.dataset_path, "test"))
    else:
        train_dataset = load_dataset(
            "json",
            data_files=os.path.join(script_args.dataset_path, "train_dataset.json"),
            split="train",
        )
        test_dataset = load_dataset(
            "json",
            data_files=os.path.join(script_args.dataset_path, "test_dataset.json"),
            split="train",
        )

    # define tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    def template_dataset(examples):
        return {
            "text":  tokenizer.apply_chat_template(
                        examples["messages"], 
                        tokenize=False, 
                        max_length=script_args.max_seq_length,
            )
        }
    
    train_dataset = train_dataset.map(template_dataset, remove_columns=["messages"])
    test_dataset = test_dataset.map(template_dataset, remove_columns=["messages"])
    
    # define model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=script_args.load_in_4bit,
            bnb_4bit_use_double_quant=script_args.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=script_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, script_args.bnb_4bit_compute_dtype),
            bnb_4bit_quant_storage=getattr(torch, script_args.bnb_4bit_quant_storage),
        ),
        attn_implementation=script_args.attn_implementation,
        torch_dtype=getattr(torch, script_args.bnb_4bit_quant_storage),
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        
    # set training arguments
    training_arguments = TrainingArguments(
        output_dir=training_args.output_dir,
        num_train_epochs=training_args.num_train_epochs,
        max_steps=training_args.max_steps,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        gradient_checkpointing_kwargs={"use_reentrant":False},
        gradient_checkpointing=training_args.gradient_checkpointing,
        bf16=training_args.bf16,
        tf32=training_args.tf32,
        max_grad_norm=training_args.max_grad_norm,
        weight_decay=training_args.weight_decay,    
        optim=training_args.optim,
        learning_rate=training_args.learning_rate,
        warmup_ratio=training_args.warmup_ratio,
        lr_scheduler_type=training_args.lr_scheduler_type,
        save_strategy=training_args.save_strategy,
        logging_steps=training_args.logging_steps,
        logging_strategy=training_args.logging_strategy,
        group_by_length=training_args.group_by_length,
        fsdp="full_shard auto_wrap offload",
        fsdp_config={
            "backward_prefetch": "backward_pre",
            "forward_prefetch": "false",
            "use_orig_params": "false",
        },
    )
    
    # define peft_config (qlora)
    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=find_all_linear_names(model), # [ "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=script_args.task_type,
    )
    
    # define trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        dataset_text_field="text",
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=peft_config,
        max_seq_length=script_args.max_seq_length,
        packing=script_args.packing,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        }
    )
    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()
    
    # begin training
    trainer.train()
    
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    
    if script_args.merge_weights_and_save:
        print(f"pid: {os.getpid()}, rank: {os.environ['LOCAL_RANK']}")
        
        # 0. save the model at output_dir path
        trainer.model.save_pretrained(training_args.output_dir, safe_serialization=False)
        
        if trainer.accelerator.is_main_process:
            # 1. clear memory
            del model
            del trainer
            torch.cuda.empty_cache()
            gc.collect()
            
            # 2. load PEFT model in fp16
            model = AutoPeftModelForCausalLM.from_pretrained(
                training_args.output_dir,
                device_map="auto",
                # torch_dtype=torch.float16,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            
            # 3. merge adapter weights with base model
            merged_model = model.merge_and_unload()
            
            merged_model.save_pretrained(script_args.sm_save_model_dir, safe_serialization=True, max_shard_size="2GB")
    else:
        trainer.model.save_pretrained(script_args.sm_save_model_dir, safe_serialization=True)
        
        # save tokenizer for easy inference
        if trainer.accelerator.is_main_process:
            if script_args.save_tokenizer:
                tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, trust_remote_code=True)
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "right"
                tokenizer.save_pretrained(script_args.sm_save_model_dir)
                print(f"end: tokenizer saved")
    


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()
    
    if script_args.finetune_with_sm:
        import sys, subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flash-attn", "--no-build-isolation"])

    # launch supervised finetuning
    set_seed(training_args.seed)
    sft(script_args, training_args)
    