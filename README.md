# KLUE-Relation Extraction

## 1️⃣ Introduction

KLUE-Relation Extraction is a task that predicts the attributes and relationships of words (entities) in a sentence. 
- **input:** sentence, subject_entity, object_entity
- **output:** pred_label, which is one of the 30 relation classes, the predicted probabilities (probs) for each of the 30 classes
- **evaluation metrics**
  - Micro F1 score, excluding the no_relation class
  - Area Under the Precision-Recall Curve (AUPRC) for all classes
  - The task is evaluated using these two metrics, with the micro F1 score being the primary metric prioritized.

## 2️⃣ What's new
This repository provides a modified version of the baseline code called dev_hf, which is based on the Hugging Face API, and a template called main based on the PyTorch Lightning API. The template utilizes the Lightning trainer and supports the following features:

- k-fold cross-validation
- Entity marker
- Syllable tokenizer
- Task-Adaptive Pre-Training (TAPT) for Masked Language Modeling (MLM)
- WandB logger
- Ensemble methods (logit/probability ensembling)
- Confusion matrix

These features are supported in the template, providing additional functionality for the task. 
## 3️⃣ Team

김별희|이원재|이정아|임성근|정준녕|
:-:|:-:|:-:|:-:|:-:
<img src='https://avatars.githubusercontent.com/u/42535803?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/61496071?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/65378914?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/14817039?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/51015187?v=4' height=80 width=80px></img>
[Github](https://github.com/kimbyeolhee)|[Github](https://github.com/wjlee-ling)|[Github](https://github.com/jjeongah)|[Github](https://github.com/lim4349)|[Github](https://github.com/ezez-refer)

## 4️⃣ Data
```
Example)
sentence: 오라클(구 썬 마이크로시스템즈)에서 제공하는 자바 가상 머신 말고도 각 운영 체제 개발사가 제공하는 자바 가상 머신 및 오픈소스로 개발된 구형 버전의 온전한 자바 VM도 있으며, GNU의 GCJ나 아파치 소프트웨어 재단(ASF: Apache Software Foundation)의 하모니(Harmony)와 같은 아직은 완전하지 않지만 지속적인 오픈 소스 자바 가상 머신도 존재한다.
subject_entity: 썬 마이크로시스템즈
object_entity: 오라클

relation: 단체:별칭 (org:alternate_names)
```
- train.csv: 32470개 <br>
- test_data.csv: 총 7765개 (정답 라벨 blind = 100으로 임의 표현) <br>

## 5️⃣ Model
<details>
    <summary><b><font size="10">Project Tree</font></b></summary>
<div markdown="1">

```
.
├─ ensemble.py
├─ inference.py
├─ main.py
├─ mlm.py
├─ model
│  ├─ __init__.py
│  ├─ loss.py
│  └─ model.py
├─ requirements.txt
├─ train.py
└─ utils
   ├─ logging.py
   ├─ make_txt.py
   └─ utils.py
```
</div>
</details>

## 6️⃣ How to Run
### Virtual Environment
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the required libraries
pip install -r requirements.txt
```

### Training
```bash
python main.py -c custom_config # using the ./config/custom_config.yaml file
python main.py -m t -c custom_config
python main.py --mode train --config custom_config
```

### Additional Training
To perform additional training, add the path to the existing model checkpoint in the `config.path.resume_path` parameter (similar to the previous commands).

```bash
python main.py -m t -c custom_config
```

### Inference
```bash
# Generates submission.csv in the prediction folder
python main.py -m i -s "saved_models/klue/bert-base.ckpt"
python main.py -m i -s "saved_models/klue/bert-base.ckpt" -c custom_config
```

### Training + Inference (Additional)
You can perform training and inference in a single run. Provide the path to the existing model checkpoint in the `config.path.resume_path` parameter and run the following command to perform additional training and inference.

```bash
python main.py --mode all --config custom_config 
python main.py -m a -c custom_config
```

### Ensemble
```bash
# Fill in the ckpt_paths (for logit ensembling) or csv_paths (for probability ensembling) in the ensemble section of the config.yaml file and run the following command
python main.py --mode ensemble 
```

### base_config.yaml
Setting `tokenizer - syllable: True` enables the syllable-level tokenizer.

## 7️⃣ Etc
`dict_label_to_num.pkl` is a pickle file that contains a dictionary mapping string labels to numeric labels for the 30 classes. Please make sure to use this dictionary to align the labels for evaluation.

```
with open('./dict_label_to_num.pkl', 'rb') as f:
    label_type = pickle.load(f)

{'no_relation': 0, 'org:top_members/employees': 1, 'org:members': 2, 'org:product': 3, 'per:title': 4, 'org:alternate_names': 5, 'per:employee_of': 6, 'org:place_of_headquarters': 7, 'per:product': 8, 'org:number_of_employees/members': 9, 'per:children': 10, 'per:place_of_residence': 11, 'per:alternate_names': 12, 'per:other_family': 13, 'per:colleagues': 14, 'per:origin': 15, 'per:siblings': 16, 'per:spouse': 17, 'org:founded': 18, 'org:political/religious_affiliation': 19, 'org:member_of': 20, 'per:parents': 21, 'org:dissolved': 22, 'per:schools_attended': 23, 'per:date_of_death': 24, 'per:date_of_birth': 25, 'per:place_of_birth': 26, 'per:place_of_death': 27, 'org:founded_by': 28, 'per:religion': 29}
```

30 class
![class](https://user-images.githubusercontent.com/65378914/217735779-266b91ec-b41f-4c47-addd-8a9174531aac.png)

