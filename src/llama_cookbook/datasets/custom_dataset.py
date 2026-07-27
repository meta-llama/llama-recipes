import importlib.machinery
import importlib.util
from pathlib import Path

DEFAULT_DATASET_FUNC = "get_custom_dataset"
DEFAULT_COLLATOR_FUNC = "get_data_collator"

def load_module_from_py_file(py_file: str) -> object:
    """
    This method loads a module from a py file which is not in the Python path
    """
    module_name = Path(py_file).name
    loader = importlib.machinery.SourceFileLoader(module_name, py_file)
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)

    loader.exec_module(module)

    return module


def parse_dataset_file(file_spec: str, default_func: str = None):
    """
    Split a `--custom_dataset.file` value into a module path and a function name.

    The optional `:func_name` suffix only names the function to load, so we split
    on the last colon to leave Windows drive letters (`C:\\data\\ds.py`) intact,
    and treat a spec that already ends in `.py` as carrying no function name.
    """
    if file_spec.endswith(".py"):
        module_path, func_name = file_spec, default_func
    else:
        module_path, _, func_name = file_spec.rpartition(":")
        if not module_path:
            module_path, func_name = file_spec, default_func

    if not module_path.endswith(".py"):
        raise ValueError(f"Dataset file {module_path} is not a .py file.")

    module_path = Path(module_path)
    if not module_path.is_file():
        raise FileNotFoundError(f"Dataset py file {module_path.as_posix()} does not exist or is not a file.")

    return module_path, func_name


def get_custom_dataset(dataset_config, tokenizer, split: str):
    module_path, func_name = parse_dataset_file(dataset_config.file, DEFAULT_DATASET_FUNC)

    module = load_module_from_py_file(module_path.as_posix())
    try:
        return getattr(module, func_name)(dataset_config, tokenizer, split)
    except AttributeError as e:
        print(f"It seems like the given method name ({func_name}) is not present in the dataset .py file ({module_path.as_posix()}).")
        raise e

def get_data_collator(dataset_processer,dataset_config):
    # The `:func_name` suffix names the dataset function only, so the collator is
    # always looked up under its own name rather than reusing that suffix.
    module_path, _ = parse_dataset_file(dataset_config.file)

    module = load_module_from_py_file(module_path.as_posix())
    collator = getattr(module, DEFAULT_COLLATOR_FUNC, None)
    if collator is None:
        print(f"Can not find the custom data_collator in the dataset.py file ({module_path.as_posix()}).")
        print("Using the default data_collator instead.")
        return None
    return collator(dataset_processer)
