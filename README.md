# ComfyUI-Qwen3_VQA_enhanced
对原始的qwen3_VQA节点的增强。其他功能保持不变，只增加了自动扫描prompt_generator文件夹功能。不新建节点组，只对原节点组进行增加与修改。

Enhancement of the original qwen3_VQA node. Other functions remain unchanged, with only the automatic scanning of the prompt_generator folder added. No new node group is created; only additions and modifications are made to the original node group.

将两个*.py文件直接下载并拷贝入原：ComfyUI\custom_nodes\ComfyUI_Qwen3-VL-Instruct文件夹中，对原始文件进行替换即可。
Directly download the two *.py files and copy them into the original: ComfyUI\custom_nodes\ComfyUI_Qwen3-VL-Instruct folder, replacing the original files.

之后即可下载qwen vl模型到\ComfyUI\models\prompt_generator文件夹中（注意只能使用qwen3_vl系列的模型）。
You can then download the Qwen VL model to the \ComfyUI\models\prompt_generator folder (note that only models from the Qwen3_vl series can be used).
<img width="611" height="358" alt="image" src="https://github.com/user-attachments/assets/49d65fe4-8f29-4836-8f01-e641e7d9b4e2" />

启动ComfyUI，使用本节点，下载的模型即出现在模型选择列表中。
Launch ComfyUI, and the downloaded model will appear in the model selection list using this node.
<img width="954" height="783" alt="image" src="https://github.com/user-attachments/assets/0484b67c-db16-47d2-a3ce-61069ca3da2b" />
