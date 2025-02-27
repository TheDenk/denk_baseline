## DENK BASELINE
  
Common scripts for fast neural network training for the tasks of semantic segmentation and classification. 

This repo combines approaches of <a href="https://github.com/Lightning-AI/lightning">PytorchLightning</a> with models from <a href="https://github.com/qubvel/segmentation_models.pytorch">Segmentation Models Pytorch</a> and <a href="https://github.com/rwightman/pytorch-image-models">timm</a>.  

Initially trained models and logs are saved in ```./output/<project folder>/<experiment name>``` 

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

#### 3. Set path to images and masks

In config file (in configs folder) you should set path to images and masks folders.   
Also you can change come parameters of training.

#### 4. Run training

Binary segmentation task:  
```python
python run.py --config "path to config.yaml"
```

  
##### Watch tensorboard (if available)

```pyhton
tensorboard --logdir=output/
```

## Contacts
<p>Issues should be raised directly in the repository. For professional support and recommendations please <a>welcomedenk@gmail.com</a>.</p>
