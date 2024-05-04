import importlib
import sys

module_name = "spatiotemporal"
module_location = "../__init__.py"

spec = importlib.util.spec_from_file_location(module_name, module_location)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)