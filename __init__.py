from .nodes import Qwen3_VQA
from .util_nodes import ImageLoader, VideoLoader
from .path_nodes import MultiplePathsInput
from .Qwen3_VQA_enhanced import Qwen3_VQA_Enhanced
WEB_DIRECTORY = "./web"
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Qwen3_VQA": Qwen3_VQA,
    "ImageLoader": ImageLoader,
    "VideoLoader": VideoLoader,
    "MultiplePathsInput": MultiplePathsInput,
    "Qwen3_VQA_Enhanced": Qwen3_VQA_Enhanced, 
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3_VQA": "Qwen3 VQA",
    "ImageLoader": "Load Image Advanced",
    "VideoLoader": "Load Video Advanced",
    "MultiplePathsInput": "Multiple Paths Input",
    "Qwen3_VQA_Enhanced": "Qwen3 VL Enhanced (Auto-Scan)"
}
