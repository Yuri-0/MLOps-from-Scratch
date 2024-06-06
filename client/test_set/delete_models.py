"""
Notes
-----
All logging functions are not return anythings, when it processed successfully (i. e. return is None).
Therefore, for confirm the functions are work well or not you can call each function, and check it return something
or not.

Here, however we can't verify that the important metadata is stored in the model registry, even though it
seems to work fine. This SDK support supplying metadata only after 'model_log' successfully completed.
If you don't process 'model_log' and you want to see the submitted metadata, you have to handle the database manually.
"""

import sys
sys.path.insert(0, '..')
from model_manager_sdk import ModelManager

# Declare endpoint of model registry and who use it
m_manager = ModelManager(endpoint='http://localhost:8000', user_id='your_id')

# Please modify the metric_name, user_id and version freely
# 'delete_model' only work when expected model already stored in the model registry
# please check the model exist the model registry or not first
m_manager.delete_models(model_name='your_model', user_id='your_id', version=0)
