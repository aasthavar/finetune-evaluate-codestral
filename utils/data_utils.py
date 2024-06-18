import pandas as pd
from datasets import (
    load_dataset, Dataset, DatasetDict
)
from .prompts import *

def create_conversation(sample, format_type="chat"):
    # a. construct prompt
    tests = sample["private_tests"]
    tests_formatted = "\n".join([
        tests_item_format.format(idx=idx, inputs=i.strip(), outputs=o.strip()) 
        for idx, (i,o) in enumerate(
            zip(tests["input"], tests["output"])
        )
    ])
    system_message = system_prompt.format(
        description=sample["description"].replace("<image>", "IMAGE"),
        tests=tests_formatted
    )
    instruction = f"{system_message}\n\n{human_prompt}"
    
    # b. construct completion
    completion = assistant_prompt.format(
        code=sample["python_solution"]
    )
    
    # c. format messages
    if format_type == "chat": # chat format
        sample["messages"] = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": human_prompt},
            {"role": "assistant", "content": completion}
        ]
    else:  # instruction format
        # sample["prompt"] = instruction[:2000] # TODO: to solve OOM -> will fix later
        sample["prompt"] = instruction
        sample["completion"] = completion
    
    return sample

def count_python_solutions(sample):
    df = pd.DataFrame(sample["solutions"])
    df_python = df[(df.language==3) | (df.language==1)]
    return df_python.shape[0]


# TODO: rethink how to do this: next idea to explore: flatten col: solutions -> then use map()
def augment_dataset(dataset):
    df = dataset.to_pandas()
    aug_rows = []
    for i, item in df.iterrows():
        for j, soln in enumerate(item["solutions"]["solution"]):
            language = item["solutions"]["language"][j]
            if (language==3 or language==1): # python3 or python2
                item_new = item.copy(deep=True)
                item_new["python_solution"] = soln
                item_new.drop('solutions', inplace=True)
                aug_rows.append(item_new)
    aug_df = pd.DataFrame(aug_rows)
    aug_ds = Dataset.from_pandas(aug_df)
    return aug_ds


def load_and_process(dataset_id, split="train[:1%]"):
    # 1. download dataset from hub
    dataset = load_dataset(dataset_id, split=split)
    
    # 2. filter - instances with 2000+ rating and contains solutions in python
    dataset = dataset.filter(lambda sample: (sample["cf_rating"] >= 2000) & (count_python_solutions(sample) >= 1))
    
    # 3. augment dataset: 1{1_problem + n_solutions} to n{1_problem + 1_solution}
    # TODO: needs code optimization
    dataset = augment_dataset(dataset) 
    
    # 4. convert dataset to instruct prompt template
    columns_to_remove = list(dataset.features)
    dataset = dataset.map(create_conversation, remove_columns=columns_to_remove, batched=False)
 
    return dataset
