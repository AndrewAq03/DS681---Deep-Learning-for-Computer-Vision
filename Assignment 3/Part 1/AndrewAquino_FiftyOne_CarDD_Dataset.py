#Andrew Aquino Part 1
#Showing the Dataset

import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub
from huggingface_hub import login
from fiftyone.core.odm.utils import load_dataset
from datasets import load_dataset




dataset = fo.Dataset.from_dir(
    dataset_dir='/workspaces/eng-ai-agents/assignments/assignment-3/CarDD',
    dataset_type=fo.types.FiftyOneDataset
    #fo_dataset_name="CarDD_local"
)

session = fo.launch_app(dataset)
session.wait()