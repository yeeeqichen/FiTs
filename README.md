## FiTs: Fine-grained Two-stage Training for Knowledge Base Question Answering

PyTorch Implementation of paper:

> **FiTs: Fine-grained Two-stage Training for Knowledge Base Question Answering (AAAI 2023)**
> 
> Bowen Cao\*, Qichen Ye\*, Nuo Chen, Weiyuan Xu, Yuexian Zou. (\* denotes equal contribution)

### 1. Download data

Download all the raw data -- ConceptNet, CommonsenseQA, OpenBookQA -- by
```
bash ./scripts/download_raw_data.sh
```

You can preprocess the raw data by running
```
CUDA_VISIBLE_DEVICES=0 python preprocess.py -p <num_processes>
```
You can specify the GPU you want to use in the beginning of the command `CUDA_VISIBLE_DEVICES=...`. The script will:
* Setup ConceptNet (e.g., extract English relations from ConceptNet, merge the original 42 relation types into 17 types)
* Convert the QA datasets into .jsonl files (e.g., stored in `data/csqa/statement/`)
* Identify all mentioned concepts in the questions and answers
* Extract subgraphs for each q-a pair

**Add MedQA-USMLE**. Besides the commonsense QA datasets (*CommonsenseQA*, *OpenBookQA*) with the ConceptNet knowledge graph, we added a biomedical QA dataset ([*MedQA-USMLE*](https://github.com/jind11/MedQA)) with a biomedical knowledge graph based on Disease Database and DrugBank. You can download all the data for this from [[here]](https://drive.google.com/file/d/1EqbiNt2ACXVrc9gmoXnzTEo9GJTe9Uor/view?usp=sharing). Unzip it and put the `medqa_usmle` and `ddb` folders inside the `data/` directory.

### 2. Post-training

```shell
bash scripts/run_post_train.sh
```
### 3. Fine-tuning
```shell
bash scripts/run_finetune.sh
```

## Reference
If you use FiTs in a research paper, please cite our work as follows:
> TODO

## Acknowledgment
This repo is built upon the following work:
```
GreaseLM: Graph REASoning Enhanced Language Models  
https://github.com/XikunZhang/greaselm
```
Many thanks to the authors and developers!