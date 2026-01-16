## DENK BASELINE
  
Common scripts for fast neural network training for classification and segmentation. 

This repo combines approaches of <a href="https://github.com/Lightning-AI/lightning">PytorchLightning</a> with models from <a href="https://github.com/qubvel/segmentation_models.pytorch">Segmentation Models Pytorch</a> and <a href="https://github.com/rwightman/pytorch-image-models">timm</a>.  

Trained models and logs are saved in `./outputs/<project>/<experiment>` by default (see `general.save_dir` in configs).

### Basic usage

#### 1. Clone repo
```python
git clone https://github.com/TheDenk/denk_baseline.git
cd denk_baseline
```

#### 2. Install requirements
```python
pip install -r requirements.txt
```

#### 3. Pick a config and set data paths

Configs live inside each project folder (recommended):

- `projects/*/config*.yaml` for project-specific runs

The root `configs/` directory contains baseline examples and is optional.  
Update dataset paths (e.g., `images_dir`, `masks_dir`, `csv_path`, `video_folder`) to match your local data layout.

#### 4. Run training

```python
python run.py --config "path/to/config.yaml"
```

To run evaluation only:

```python
python run.py --config "path/to/config.yaml" --test
```

  
##### Watch tensorboard (if available)

```python
tensorboard --logdir=outputs/
```

## Contacts
<p>Issues should be raised directly in the repository. For professional support and recommendations please <a>welcomedenk@gmail.com</a>.</p>
