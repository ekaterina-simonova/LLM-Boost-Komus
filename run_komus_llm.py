import os

 

os.environ["HF_HOME"] = r"D:\LLM-Boost-CatBoost\hf_cache"


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print("HF_HOME =", os.environ["HF_HOME"])
 

from Tablet import evaluate

 

benchmark_path = "./benchmark/performance"

 

MODEL_NAME = r"D:\LLM-Boost-CatBoost\models\Qwen3-8B"

DATASET_NAME = "KomusLLMBoostSmoke"

 

k_shot = 3

seed = 42

 

if "flan" in MODEL_NAME.lower():

    model_family = "flan"

elif "llama" in MODEL_NAME.lower() or "qwen" in MODEL_NAME.lower():

    model_family = "llama"

else:

    raise ValueError(f"Unsupported model: {MODEL_NAME}")

 

task = [

    DATASET_NAME + "/prototypes-naturallanguage-performance-0"

]

 

save_name = "Qwen3-8B"

 

save_paths = [

    DATASET_NAME + f"_{save_name}_{k_shot}-shot_{seed}"

]

 

evaluator = evaluate.Evaluator(

    benchmark_path=benchmark_path,

    tasks_to_run=task,

    model=MODEL_NAME,

    encoding_format=model_family,

    results_file=save_name + ".txt",

    k_shot=k_shot,

    save_paths=save_paths,

)

 

print("Starting LLM inference...")

print("Model:", MODEL_NAME)

print("Task:", task[0])

print("k-shot:", k_shot)

 

evaluator.run_eval(

    how_many=1,

    seed=seed

)

 

print("LLM inference finished.")