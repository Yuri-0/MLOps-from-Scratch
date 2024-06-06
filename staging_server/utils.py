import os
import re
import sqlite3


def install(root_path):
    os.makedirs(os.path.join(root_path, 'model_registry', 'repository'))

    with sqlite3.connect(os.path.join(root_path, 'model_registry', 'storage.db'), isolation_level=None) as conn:
        cursor = conn.cursor()

        cursor.execute(
            'CREATE TABLE experiment '
            '(experiment_id, name, user_id, status, start_time, end_time, artifact_uri, model_size, lifecycle_stage, deleted_time)'
        )

        cursor.execute(
            'CREATE TABLE dataset '
            '(dataset_name, dataset_uri, dataset_type, experiment_id)',
        )

        cursor.execute(
            'CREATE TABLE metric '
            '(metric_name, value, timestamp, experiment_id, step)',
        )

        cursor.execute(
            'CREATE TABLE params '
            '(param_name, value, experiment_id)',
        )

        cursor.execute(
            'CREATE TABLE tags '
            '(tags, value, experiment_id)',
        )

        cursor.execute(
            'CREATE TABLE resource '
            '(cpu, nvidia_gpu, custom_resource, experiment_id, system, machine)',
        )


def auto_tagging(requirements):
    patterns = {
        'sklearn_version': re.compile(r'scikit-learn==(.*?)\n'),
        'tensorflow_version': re.compile(r'tensorflow==(.*?)\n'),
        'tensorflow-gpu_version': re.compile(r'tensorflow-gpu==(.*?)\n'),
        'pytorch_version': re.compile(r'torch==(.*?)\n'),
        'torch-vision_version': re.compile(r'torchvision==(.*?)\n'),
        'torch-audio_version': re.compile(r'torchaudio==(.*?)\n')
    }

    tags = {}
    for tag_name, pattern in patterns.items():
        result = pattern.search(requirements)
        if result:
            tags[tag_name] = result.group(1)

    return tags
