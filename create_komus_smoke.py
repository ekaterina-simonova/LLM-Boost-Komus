import os

import pandas as pd

from Tablet import create

 

NAME = "KomusDefaultSmoke"

 

HEADER = (

    "Given financial and qualitative information about a company, "

    "predict whether the company will default within one year."

)

 

DATA_PATH = os.path.join(

    "data",

    NAME,

    "prototypes-synthetic-performance-0"

)

 

train = pd.read_csv(

    os.path.join(DATA_PATH, "train.csv"),

    index_col=0

)

 

test = pd.read_csv(

    os.path.join(DATA_PATH, "test.csv"),

    index_col=0

)

 

train_y = train.pop("y_temp")

test_y = test.pop("y_temp")

 

assert "INN" not in train.columns

assert "Q_B1_norm" not in train.columns

assert "Q_B2_norm" not in train.columns

 

print("Train:", train.shape)

print("Test:", test.shape)

print("Production features:", train.shape[1])

 

create.create_task(

    train_df=train,

    test_df=test,

    train_y=train_y.to_numpy(),

    test_y=test_y.to_numpy(),

    name=NAME,

    header=HEADER,

    categorical_columns=[],

    save_loc="./benchmark",

    nl_instruction=HEADER,

    experiment_name="performance",

    num=0,

    temp_y_col_name="y_temp",

    only_natural_language=True,

    seed=42,

)

 

print("Smoke TABLET task created successfully.")