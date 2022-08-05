## SEGMENTATION BASELINE
  
Common scripts for fast neural network training for the task of binary and multiclass semantic segmentation. 

This repo combines approaches of <a href="https://github.com/Lightning-AI/lightning">PytorchLightning</a> and <a href="https://github.com/qubvel/segmentation_models.pytorch">Segmentation Models Pytorch</a> for models training.  

Initially trained models and logs are saved in ```./output/project folder/experiment name``` 

### Basic usage

#### 1. Clone repo
```python
git clone https://github.com/TheDenk/segmentation_baseline.git
cd segmentation_baseline
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
python main.py --config ./configs/example_binary_config.yaml
```

Multiclass segmentation task:  
```python
python main.py --config ./configs/example_multiclass_config.yaml
```
  
##### Watch tensorboard

```pyhton
tensorboard --logdir=output/
```

## Contacts
<p>Issues should be raised directly in the repository. For professional support and recommendations please <a>welcomedenk@gmail.com</a>.</p>