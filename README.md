# Graph Transformer via ResNet SimCLR for Whole Slide Image (WSI) Colon Classification

### Implementation from https://www.medrxiv.org/content/10.1101/2021.10.15.21265060v1

## 1. Training a Patch Feature Extractor
 - cd feature_extractor/
 - python feature_extract.py (check config.yaml)
 
## 2. Build Graph
 - python build_graphs.py (check arg parser)
 
## 3. Training Graph Transformer
 - cd ..
 - python main.py (check arg parser at option.py)


## 4. Visualization (In Progress)
