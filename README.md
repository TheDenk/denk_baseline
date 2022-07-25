## SEGMENTATION BASELINE
  
Base scripts to quickly train neural network for binary or multiclass semantic segmentation task.  

This repo unions scripts of <a href="">PytorchLightning</a> and <a href="">Segmentation Models Pytorch</a> to train models.  

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

In config files (in configs folder) you should set path to images and masks folders.   
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
tensorboard --logdir=lightning_logs/
```