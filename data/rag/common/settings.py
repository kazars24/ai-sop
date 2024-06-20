import json

class SingletonConfig:
    _instance = None

    def __new__(cls, config_name='ollama_phi-3-medium-128k'):
        if cls._instance is None:
            with open(f'data/rag/common/configs/{config_name}.json', 'r') as file:
                cls._instance = json.load(file)
        return cls._instance
