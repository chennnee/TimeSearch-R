# __init__.py

REWARD_MODULES = {
    "v1":   (".v1",   "reward_functions", "reward_weights"),
    "v2":   (".v2",   "reward_functions", "reward_weights"),
    "v3":   (".v3",   "reward_functions", "reward_weights"),
    "v3_1": (".v3_1", "reward_functions", "reward_weights"),
    "v4":   (".v4",   "reward_functions", "reward_weights"),
    "v5":   (".v5",   "reward_functions", "reward_weights"),
    "v5_1": (".v5_1", "reward_functions", "reward_weights"),
    "v6":   (".v6",   "reward_functions", "reward_weights"),
    "v7":   (".v7",   "reward_functions", "reward_weights"),
}

def get_reward_functions(version: str):
    if version not in REWARD_MODULES:
        raise ValueError(f"Invalid reward version: {version}")
    
    module_path, func_name, weight_name = REWARD_MODULES[version]
    
    # 只在真正需要时才 import 对应版本
    import importlib
    mod = importlib.import_module(module_path, package=__package__)
    
    return getattr(mod, func_name), getattr(mod, weight_name)