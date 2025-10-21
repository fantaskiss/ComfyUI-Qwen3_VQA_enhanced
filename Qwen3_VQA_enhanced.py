import os
import torch
import folder_paths
from torchvision.transforms import ToPILImage
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
import model_management
from qwen_vl_utils import process_vision_info
from pathlib import Path


class Qwen3_VQA_Enhanced:
    def __init__(self):
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        self.device = model_management.get_torch_device()
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )
        self.available_models = self.scan_available_models()

    def scan_available_models(self):
        """扫描 prompt_generator 目录，获取可用的模型列表"""
        models_dir = os.path.join(folder_paths.models_dir, "prompt_generator")
        available_models = []
        
        # 预定义的标准模型列表（作为备选）
        standard_models = [
            "Qwen3-VL-4B-Instruct-FP8",
            "Qwen3-VL-4B-Thinking-FP8", 
            "Qwen3-VL-8B-Instruct-FP8",
            "Qwen3-VL-8B-Thinking-FP8",
            "Qwen3-VL-4B-Instruct",
            "Qwen3-VL-4B-Thinking",
            "Qwen3-VL-8B-Instruct",
            "Qwen3-VL-8B-Thinking",
        ]
        
        # 如果目录不存在，返回标准列表
        if not os.path.exists(models_dir):
            print(f"Info: Models directory {models_dir} does not exist. Using standard model list.")
            return standard_models
        
        try:
            # 扫描目录中的模型文件夹
            for item in os.listdir(models_dir):
                item_path = os.path.join(models_dir, item)
                if os.path.isdir(item_path):
                    # 检查是否包含必要的模型文件
                    required_files = ["config.json", "model.safetensors", "model.safetensors.index.json"]
                    has_required_files = any(
                        os.path.exists(os.path.join(item_path, f)) for f in required_files
                    )
                    
                    if has_required_files:
                        available_models.append(item)
            
            # 如果找到了本地模型，优先使用本地模型
            if available_models:
                print(f"Found local models: {available_models}")
                # 合并本地模型和标准模型，去除重复，保持排序
                all_models = list(set(available_models + standard_models))
                return sorted(all_models)
            else:
                print("No local models found. Using standard model list.")
                return standard_models
                
        except Exception as e:
            print(f"Error scanning models directory: {e}. Using standard model list.")
            return standard_models

    @classmethod
    def INPUT_TYPES(s):
        # 创建实例来获取可用的模型列表
        instance = s()
        available_models = instance.available_models
        default_model = available_models[0] if available_models else "Qwen3-VL-4B-Instruct-FP8"
        
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    available_models,
                    {"default": default_model},
                ),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "none"},
                ),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 2048, "min": 128, "max": 2048, "step": 1},
                ),
                "min_pixels": (
                    "INT",
                    {
                        "default": 256 * 28 * 28,
                        "min": 4 * 28 * 28,
                        "max": 16384 * 28 * 28,
                        "step": 28 * 28,
                    },
                ),
                "max_pixels": (
                    "INT",
                    {
                        "default": 1280 * 28 * 28,
                        "min": 4 * 28 * 28,
                        "max": 16384 * 28 * 28,
                        "step": 28 * 28,
                    },
                ),
                "seed": ("INT", {"default": -1}),
                "attention": (
                    [
                        "eager",
                        "sdpa",
                        "flash_attention_2",
                    ],
                    {"default": "eager"},
                ),
            },
            "optional": {"source_path": ("PATH",), "image": ("IMAGE",)},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_Qwen3-VL-Instruct_Enhanced"  # 修改分类名称以示区别

    def inference(
        self,
        text,
        model,
        keep_model_loaded,
        temperature,
        max_new_tokens,
        min_pixels,
        max_pixels,
        seed,
        quantization,
        source_path=None,
        image=None,
        attention="eager",
    ):
        if seed != -1:
            torch.manual_seed(seed)
        
        # 构建模型ID - 保持与原节点相同的逻辑
        model_id = f"qwen/{model}"
        self.model_checkpoint = os.path.join(
            folder_paths.models_dir, "prompt_generator", os.path.basename(model_id)
        )

        # 如果模型不存在则下载 - 保持与原节点相同的逻辑
        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=model_id,
                local_dir=self.model_checkpoint,
                local_dir_use_symlinks=False,
            )

        # 处理器初始化 - 保持与原节点相同的逻辑
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                self.model_checkpoint, min_pixels=min_pixels, max_pixels=max_pixels
            )

        # 模型加载 - 保持与原节点相同的逻辑
        if self.model is None:
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                )
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                quantization_config = None

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_checkpoint,
                dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
                attn_implementation=attention,
                quantization_config=quantization_config,
            )

        # 图像处理 - 保持与原节点相同的逻辑
        temp_path = None
        if image is not None:
            pil_image = ToPILImage()(image[0].permute(2, 0, 1))
            temp_path = Path(folder_paths.temp_directory) / f"temp_image_{seed}.png"
            pil_image.save(temp_path)

        # 推理过程 - 保持与原节点相同的逻辑
        with torch.no_grad():
            if source_path:
                messages = [
                    {
                        "role": "system",
                        "content": "You are QwenVL, you are a helpful assistant expert in turning images into words.",
                    },
                    {
                        "role": "user",
                        "content": source_path
                        + [
                            {"type": "text", "text": text},
                        ],
                    },
                ]
            elif temp_path:
                messages = [
                    {
                        "role": "system",
                        "content": "You are QwenVL, you are a helpful assistant expert in turning images into words.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{temp_path}"},
                            {"type": "text", "text": text},
                        ],
                    },
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                        ],
                    }
                ]

            # 准备推理输入 - 保持与原节点相同的逻辑
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            # 生成输出 - 保持与原节点相同的逻辑
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, temperature=temperature
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            result = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            # 内存管理 - 保持与原节点相同的逻辑
            if not keep_model_loaded:
                del self.processor
                del self.model
                self.processor = None
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

            return (result[0],) if result else ("",)


