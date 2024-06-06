import sys
sys.path.insert(0, '..')
from model_manager_sdk import ModelManager

# Declare endpoint of model registry and who use it
m_manager = ModelManager(endpoint='http://localhost:8000', user_id='your_id')

# Please modify the tag_name and tag freely
# 'list_models' not take any arguments
m_manager.list_models()
