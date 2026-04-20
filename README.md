<h1 align="center">CoRe-BT: A Multimodal Radiology-Pathology-Text Benchmark for Robust Brain Tumor Typing </h1>
<p align="center">
    <a href="https://www.imageclef.org/2026/medical/mediqa-core">Challenge Description</a> |
    <a href="https://ai4media-bench.aimultimedialab.ro/competitions/6/">Registration</a> |
    <a href="https://arxiv.org/pdf/2603.03618">Manuscript</a> |
    <a href="#citation">BibTeX</a>
    
</p>

<p align="center">
  <span style="background-color: white; padding: 20px; display: inline-block;">
    <img src="assets/corebt_figure.png" width="900">
  </span>
</p>

Juampablo E. Heras Rivera†, Daniel K Low†, Wen-wai Yim, Jacob Ruzevick, Xavier Xiong, Mehmet Kurt*, Asma Ben Abacha* 

† Equal contribution, * Shared last authorship



<div align="center">
<table>
<tr>
<td>


**[KurtLab, University of Washington](https://www.kurtlab.com/)** <br/>
**[Microsoft Health AI, Microsoft](https://www.microsoft.com/en-us/research/lab/microsoft-health-futures/)**

</td>
<!-- <td width="200"></td> spacer column -->
<td align="right">
  <img src="assets/affiliations.png" width="220" alt="BTReport affiliations">
</td>
</tr>
</table>
</div>



### Updates 
- 4/20/2026: Added [`experiments_mediqa`](experiments_mediqa) directory with code to baseline solution for [ImageCLEFmed MEDIQA-CORE 2026: Brain Tumor Subtype Classification Challenge](https://ai4media-bench.aimultimedialab.ro/competitions/6/).




![-----------------------------------------------------](assets/purpleline.png)


<h2 align="center">Benchmark Overview</h2>

### Dataset

| Component | Count | Description |
|-----------|------:|-------------|
| Patients | 310 | Glioma patients collected at the University of Washington (2023–2025) |
| MRI volumes | 310 subjects | Each with up to 4 multi-sequence brain MRIs (T1, T1c, T2, FLAIR) |
| Histopathology subjects | 95 | Patients with paired H&E whole-slide pathology images |
| Whole-slide images (WSI) | 597 slides | Gigapixel histopathology slides (~6 per subject on average) |
| Pathology reports | 95 | Free-text neuropathology diagnostic reports |
| Tumor masks | subset of MRI cases | Expert-corrected segmentation masks for edema, enhancing tumor, and necrotic core |

### Benchmark Tasks

| Task | Prediction Target | Classes | Description |
|-----|------------------|--------:|-------------|
| WHO Grade Prediction | WHO tumor grade | 3 | Predict WHO Grade II, III, or IV |
| LGG vs HGG Classification | Tumor grade group | 2 | Binary classification of low-grade versus high-grade glioma |
| Molecular Tumor Type (Level-1) | Molecular tumor grouping | 4 | Coarse tumor categories reflecting biologically meaningful molecular subtypes |

![-----------------------------------------------------](assets/purpleline.png)


<h2 align="center">Repo Structure</h2>

<table align="center">
<tr>
<td><strong><a href="./CLAM/">CLAM/</a></strong></td>
<td>
Data preprocessing and whole-slide tiling utilities based on CLAM<sup>[1]</sup>. 
Includes custom artifact removal using HSV color-based segmentation and tiling pipelines for WSI patch extraction.

<details>
<summary>Example</summary>
<img src="assets/artifact_removal_example_clam.png" width="1000">
</details>

</td>
</tr> 

<tr>
<td><strong><a href="./gigapath/">gigapath/</a></strong></td>
<td>
Whole-slide histopathology embedding pipeline using the Prov-GigaPath foundation model<sup>[2]</sup>.
Uses tiles generated from CLAM to compute slide-level embeddings.

<details>
<summary>Example</summary>
<img src="assets/tiling_pca.png" width="900">
</details>

</td>
</tr>

<tr>
<td><strong><a href="./NeuroVFM/">NeuroVFM/</a></strong></td>
<td>
MRI foundation model framework for multi-sequence brain MRI embedding<sup>[3]</sup>.
Produces subject-level embeddings from T1, T1c, T2, and FLAIR sequences.
</td>
</tr>


<tr>
<td><strong><a href="./experiments/">experiments/</a></strong></td>
<td>
Scripts for multimodal embedding fusion and downstream tumor typing experiments, including modality ablation studies and evaluation pipelines.
</td>
</tr>
</table>

<br>

<sub>
[1] Lu, M.Y., Williamson, D.F.K., Chen, T.Y. et al. Data-efficient and weakly supervised computational pathology on whole-slide images. Nat Biomed Eng 5, 555–570 (2021).  <br>
[2] Xu, Hanwen, et al. "A whole-slide foundation model for digital pathology from real-world data." Nature 630.8015 (2024): 181-188.   <br>
[3] Kondepudi, Akhil, et al. "Health system learning achieves generalist neuroimaging models." arXiv preprint arXiv:2511.18640 (2025).
</sub>

![-----------------------------------------------------](assets/purpleline.png)
<h2 align="center">Usage</h2>


## I. Setup

```bash
# Install uv 
curl -LsSf https://astral.sh/uv/install.sh | sh 

# create env by installing packages from uv.lock file, activate env
cd CoreBT/experiments_mediqa
uv sync

# verify the installation and CUDA support
uv run python -c "import torch; print(f'Torch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}')"
```


## II. Prerequisite Downloads

###  Download embeddings
Download the necessary embeddings and dataset CSVs. Ensure your CSV files contain the required headers (`subject_id` and *_label).

> **Note:** Within `dataset_csvs`, only `train.csv` contains verified labels. Other files (e.g., validation or test sets) contain placeholder random labels.

### Add modality presence columns to CSV
Update the file paths within `utils/add_presence_columns_to_csv.py` to point to your local data directories, then run to scan for available imaging modalities and update the CSVs with presence columns. 

```bash
uv run utils/add_presence_columns_to_csv.py
```

After running the script, your CSV will be structured as follows:

<div align="center">

| subject_id | level1_label | lgghgg_label | who_grade_label | histopathology_present | mri_present |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CoReBT-0037 | 3 | 0 | 0 | False | True |
| CoReBT-0038 | 2 | 1 | 1 | True | True |
| ... | ... | ... | ... | ... | ... |
</div>

For further reference, see the examples provided in the `dataset_csvs` directory.


## III. Run training

Our solution scripts are divided in three components: 
1) Histopathology-specific linear probe (`experiments_mediqa/corebt_histo_main.py`)
2) MRI-specific linear probe (`experiments_mediqa/corebt_mri_main.py`)
3) Linear probe fusion with adapter steps (`experiments_mediqa/corebt_fusion_main.py`)

All scripts can be run with `experiments_mediqa/run_corebt_linear_probes.py`

> Note: Because K-fold CV is not implemented yet, the following scripts use the `train.csv` for both training and validation. 
The code below will perform inference on the subjects in `TEST_CSV` and `TEST_MRI/TEST_HISTO`, so we pass in the validation subjects there to generate the predictions for those specific samples.


```bash
CSV_DIR=experiments_mediqa/dataset_csvs
EMBED_DIR=/path/to/corebt_dataset
OUTPUT_DIR=experiments_mediqa/runs

# Metadata CSV Paths 
TRAIN_CSV="$CSV_DIR/train.csv"
VAL_CSV="$CSV_DIR/train.csv"          
TEST_CSV="$CSV_DIR/val_randomized.csv"

# MRI Embedding Paths
TRAIN_MRI="$EMBED_DIR/MRI_Embeddings_train.zip"
VAL_MRI="$EMBED_DIR/MRI_Embeddings_train.zip"
TEST_MRI="$EMBED_DIR/MRI_Embeddings_val.zip"

# Histopathology Embedding Paths
TRAIN_HISTO="$EMBED_DIR/Pathology_Embeddings_train.zip"
VAL_HISTO="$EMBED_DIR/Pathology_Embeddings_train.zip"
TEST_HISTO="$EMBED_DIR/Pathology_Embeddings_val.zip"

# Hyperparameters 
OUTPUT_DIR="runs"
MRI_EMBED_DIM=768
HISTO_EMBED_DIM=768
BATCH_SIZE=32
TRAIN_ITERS=600
LR=0.001
MIN_LR=0.0
OPTIM="adam"
MOMENTUM=0.0
WEIGHT_DECAY=1e-4
EVAL_INTERVAL=10
NUM_WORKERS=4
SEED=42
LABEL_PREFIX="all" # choices: level1, lgghgg, who_grade, all

```



### MRI linear probe training with `corebt_mri_main.py`

```bash
uv run experiments_mediqa/corebt_mri_main.py \
    --train_csv_path "$TRAIN_CSV" \
    --val_csv_path "$VAL_CSV" \
    --test_csv_path "$TEST_CSV" \
    --train_mri_embed_path "$TRAIN_MRI" \
    --val_mri_embed_path "$VAL_MRI" \
    --test_mri_embed_path "$TEST_MRI" \
    --label_prefix $LABEL_PREFIX \
    --embed_dim $MRI_EMBED_DIM \
    --batch_size $BATCH_SIZE \
    --train_iters $TRAIN_ITERS \
    --lr $LR \
    --min_lr $MIN_LR \
    --optim $OPTIM \
    --momentum $MOMENTUM \
    --weight_decay $WEIGHT_DECAY \
    --eval_interval $EVAL_INTERVAL \
    --num_workers $NUM_WORKERS \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR/mri"

```

### Histopathology linear probe training with `corebt_histo_main.py`
```bash
uv run experiments_mediqa/corebt_histo_main.py \
    --train_csv_path "$TRAIN_CSV" \
    --val_csv_path "$VAL_CSV" \
    --test_csv_path "$TEST_CSV" \
    --train_histo_embed_path "$TRAIN_HISTO" \
    --val_histo_embed_path "$VAL_HISTO" \
    --test_histo_embed_path "$TEST_HISTO" \
    --label_prefix $LABEL_PREFIX \
    --embed_dim $HISTO_EMBED_DIM \
    --batch_size $BATCH_SIZE \
    --train_iters $TRAIN_ITERS \
    --lr $LR \
    --min_lr $MIN_LR \
    --optim $OPTIM \
    --momentum $MOMENTUM \
    --weight_decay $WEIGHT_DECAY \
    --eval_interval $EVAL_INTERVAL \
    --num_workers $NUM_WORKERS \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR/histopathology"
```


### MRI + Histopathology Fusion adapter training with `corebt_histo_main.py`

```bash
uv run experiments_mediqa/corebt_fusion_main.py \
    --train_csv_path "$TRAIN_CSV" \
    --val_csv_path "$VAL_CSV" \
    --test_csv_path "$TEST_CSV" \
    --train_mri_embed_path "$TRAIN_MRI" \
    --val_mri_embed_path "$VAL_MRI" \
    --test_mri_embed_path "$TEST_MRI" \
    --train_histo_embed_path "$TRAIN_HISTO" \
    --val_histo_embed_path "$VAL_HISTO" \
    --test_histo_embed_path "$TEST_HISTO" \
    --label_prefix $LABEL_PREFIX \
    --histo_embed_dim $HISTO_EMBED_DIM \
    --mri_embed_dim $MRI_EMBED_DIM \
    --batch_size $BATCH_SIZE \
    --train_iters $TRAIN_ITERS \
    --lr $LR \
    --min_lr $MIN_LR \
    --optim $OPTIM \
    --momentum $MOMENTUM \
    --weight_decay $WEIGHT_DECAY \
    --eval_interval $EVAL_INTERVAL \
    --num_workers $NUM_WORKERS \
    --seed $SEED \
    --output_dir "$OUTPUT_DIR/fusion"
```


### All models training with `run_corebt_linear_probes.py`
```bash
uv run experiments_mediqa/run_corebt_linear_probes.py \
    --train_csv_path "$TRAIN_CSV" \
    --val_csv_path "$VAL_CSV" \
    --test_csv_path "$TEST_CSV" \
    --train_mri_embed_path "$TRAIN_MRI" \
    --val_mri_embed_path "$VAL_MRI" \
    --test_mri_embed_path "$TEST_MRI" \
    --train_histo_embed_path "$TRAIN_HISTO" \
    --val_histo_embed_path "$VAL_HISTO" \
    --test_histo_embed_path "$TEST_HISTO" \
    --output_dir "$OUTPUT_DIR" \
    --label_prefix "$LABEL_PREFIX" \
    --variant "all" # choices=["mri", "histo", "fusion", "all"], default="all"
```


## IV. Collect predictions
Our submitted predictions are made using a combination of the predictions from the MRI probe and the Histopathology probe. The `experiments_mediqa/utils/collect_final_prediction.py` file combines the labels, taking the MRI labels as priority. Modify this with your own paths if using.

```bash
uv run experiments_mediqa/utils/collect_final_prediction.py
```

![-----------------------------------------------------](assets/purpleline.png)


<h2 align="center">Evaluation</h2>

### Command
We evaluate model predictions with following command:
```bash
python3 eval/evaluate_predictions.py \
  --submission_csv example_submission.csv \ # Path to your model's predictions csv
  --reference_csv eval/corebt_groundtruth_devset.csv \       # Path to the ground truth labels
  --task all \                               # Select one or more: [level1, lgghgg, who, all]
  --run_id my_run \                          # Unique ID for this eval run (defaults to "run")
  --output_json results.json                 # Path to save results 
```

### Inputs

The submission file provided via `--submission_csv` must be a CSV with one row per subject, see `eval/example_submission.csv` for an example, and `eval/corebt_groundtruth_devset.csv` to understand how classes are defined in the training split.

Expected structure:

<div align="center">

| subject_id | level1_pred | lgghgg_pred | who_grade_pred |
| :--- | :--- | :--- | :--- |
| U0027924 | 3 | 0 | 0 |
| U0808219 | 2 | 1 | 1 |
| ... | ... | ... | ... |
</div>

### Outputs

Results are saved in a JSON file to the path provided in `--output_json`.

```json
// results.json 
{
  "runs": {
    "my_run": {
      "tasks": {
        "level1": {
          "summary": { "num_samples": "..." },
          "global_metrics": {
            "accuracy": "...",
            "balanced_accuracy": "...",
            "f1_macro": "...",
            "f1_weighted": "...",
            "precision_macro": "...",
            "recall_macro": "..."
          },
          "per_class_metrics": {
            "0": { "support": "...", "fraction": "...", "precision": "...", "recall": "...", "f1": "..." },
            "1": { "support": "...", "fraction": "...", "precision": "...", "recall": "...", "f1": "..." },
            "2": { "..." : "..." }
          },
          "confusion_matrix": {
            "labels": [0, 1, "..."],
            "matrix": [
              [30, 29, "..."],
              ["...", "...", "..."]
            ]
          },
          "task": "level1",
          "run_id": "my_run"
        },
        "lgghgg": { "..." : "Follows level1 structure" },
        "who":    { "..." : "Follows level1 structure" }
      }
    }
  }
}
```


<h2 id="citation" align="center">Citation</h2>
    
    @inproceedings{CoRe-BT-arXiv,
    
    author       = {Juampablo E. {Heras Rivera} and 
                    Daniel K. Low and 
                    Xavier Xiong and 
                    Jacob J. Ruzevick and 
                    Daniel D. Child and 
                    Wen-wai Yim and 
                    Mehmet Kurt and 
                    Asma {Ben Abacha}}, 
    
    title        = {CoRe-BT: A Multimodal Radiology-Pathology-Text Benchmark for Robust Brain Tumor Typing}, 
    
    journal      = {CoRR}, 
    
    volume       = {abs/2603.03618},
    
    year         = {2026},
    
    url          = {https://arxiv.org/abs/2603.03618}
    }


