# Prepare datasets
before running the code, crate folder named "datasets" in the project folder with structure like:
- **flickr30k-images**: This folder contains all the images related to the Flickr 30k dataset.
- **JSON Files**:
  - `en_train.json`: Contains the English training data.
  - `en_val.json`: Contains the English validation data.

# deploy.py
after running the train.py, we have a .pth file, then run the deploy.py we will get respectively `best_epoch_weights.onnx`, `best_epoch_weights.trt`

# infer_using_trt.py
the demo code that shows how the .trt file works
