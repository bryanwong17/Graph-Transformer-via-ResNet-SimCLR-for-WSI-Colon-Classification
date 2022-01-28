# Graph-Transformer-via-ResNet-for-WSI-Colon-Classification


## 1. Training a Patch Feature Extractor
 - cd feature_extractor/
 - python feature_extract.py (check config.yaml)
 
## 2. Build Graph
 - python build_graphs.py (check arg parser)
 
## 3. Training Graph Transformer.py
 - cd ..
 - python main.py (check arg parser at option.py)