import os
from pathlib import Path
from datasets import load_dataset, ClassLabel
from PIL import Image
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 配置参数 ---
# Hugging Face 数据集名称
DATASET_NAME = "imagenet-1k"

# 本地存储路径 (相对于脚本执行位置)
BASE_OUTPUT_DIR = Path("./data")
IMAGENET_DIR = BASE_OUTPUT_DIR / "imagenet"
TRAIN_DIR = IMAGENET_DIR / "train"
VAL_DIR = IMAGENET_DIR / "val"

# Hugging Face datasets 库的缓存目录
HF_CACHE_DIR = BASE_OUTPUT_DIR / "hf_cache"

# 测试模式：只下载并保存少量样本进行测试
# NUM_SAMPLES_TO_TEST = 5 # 注释掉此行以进行完整下载
# --- ---

def save_image_split(dataset_split, target_dir: Path, class_names: list, split_name: str):
    """
    保存数据集的特定划分 (train/validation) 到目标目录。

    Args:
        dataset_split: Hugging Face 数据集的一个划分 (e.g., dataset['train']).
        target_dir: 保存该划分的根目录 (e.g., TRAIN_DIR).
        class_names: 类别名称列表 (synset IDs, e.g., ['n01440764', ...]).
        split_name: 划分的名称，用于日志和文件名 (e.g., "train").
    """
    logger.info(f"开始处理和保存 {split_name} 数据集...")
    target_dir.mkdir(parents=True, exist_ok=True)

    # 为每个类别创建子文件夹
    for synset_id in class_names:
        (target_dir / synset_id).mkdir(parents=True, exist_ok=True)

    # 记录每个类别已保存的图片数量，用于生成唯一文件名
    image_counters = {synset_id: 0 for synset_id in class_names}

    for i, example in enumerate(tqdm(dataset_split, desc=f"保存 {split_name} 图片")):
        try:
            image = example['image']
            label_idx = example['label']
            
            # 获取类别名称 (synset ID)
            synset_id = class_names[label_idx]
            
            class_specific_dir = target_dir / synset_id
            
            # 生成文件名
            image_counters[synset_id] += 1
            filename = f"{split_name}_{synset_id}_{image_counters[synset_id]:08d}.jpg"
            filepath = class_specific_dir / filename
            
            # 确保图像是 RGB 格式 (ImageNet 主要是 JPEG)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image.save(filepath, format="JPEG")

        except Exception as e:
            logger.error(f"处理 {split_name} 数据时发生错误 (索引 {i}, 标签 {example.get('label', 'N/A')}): {e}")
            logger.error(f"问题数据: {example}")
            # 可以选择跳过有问题的图片或采取其他错误处理措施

    logger.info(f"{split_name} 数据集保存完成。保存在: {target_dir}")

def main():
    # logger.info(f"开始下载和处理 ImageNet-1K 数据集 (测试模式：每个划分最多 {NUM_SAMPLES_TO_TEST} 个样本)...")
    logger.info(f"开始下载和处理完整的 ImageNet-1K 数据集...") # 更新日志信息
    logger.info(f"目标基础目录: {BASE_OUTPUT_DIR.resolve()}")
    logger.info(f"ImageNet 将存储在: {IMAGENET_DIR.resolve()}")
    logger.info(f"Hugging Face 缓存目录: {HF_CACHE_DIR.resolve()}")

    # 创建输出目录
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"正在从 Hugging Face Hub 加载 '{DATASET_NAME}' 数据集 (流式传输)...")
    logger.info("这可能需要很长时间，具体取决于你的网络和数据集大小。")
    logger.info("Hugging Face datasets 库会自动处理下载和缓存。")

    class_names = None # 初始化 class_names

    try:
        # 使用 streaming=True 加载数据集
        # 注意：'imagenet-1k' 可能需要你通过 huggingface-cli login 进行了身份验证
        # 并且可能需要你同意数据集的使用条款 (通常在 Hugging Face 网站上操作)
        dataset_stream = load_dataset(DATASET_NAME, cache_dir=str(HF_CACHE_DIR), streaming=True, trust_remote_code=True)
        logger.info("数据集流式加载尝试成功。")

        # 获取类别名称 (synset IDs) - 对于流式数据集，需要从特征获取
        label_feature_source_split = None
        if 'train' in dataset_stream and dataset_stream['train'] is not None and hasattr(dataset_stream['train'], 'features'):
            label_feature_source_split = 'train'
        elif 'validation' in dataset_stream and dataset_stream['validation'] is not None and hasattr(dataset_stream['validation'], 'features'):
            label_feature_source_split = 'validation'
        elif 'val' in dataset_stream and dataset_stream['val'] is not None and hasattr(dataset_stream['val'], 'features'):
            label_feature_source_split = 'val'

        if not label_feature_source_split:
            logger.error("在数据流的任何可用划分 ('train', 'validation', 'val') 中均未找到有效的特征信息。无法获取类别名称。")
            return

        logger.info(f"尝试从 '{label_feature_source_split}' 划分获取类别特征。")
        current_features = dataset_stream[label_feature_source_split].features
        if 'label' not in current_features:
            logger.error(f"'{label_feature_source_split}' 划分的特征中未找到 'label'。 特征: {current_features}")
            return
        
        label_feature_obj = current_features['label']
        if not isinstance(label_feature_obj, ClassLabel):
            logger.error(f"数据集的 'label' 特征不是 ClassLabel 类型 (而是 {type(label_feature_obj)})。无法自动获取类别名称。")
            return
        
        class_names = label_feature_obj.names
        if not class_names: # ClassLabel.names could be None or empty
            logger.error("从 ClassLabel 特征获取的类别名称列表为空或 None。")
            return

        logger.info(f"获取到 {len(class_names)} 个类别。前5个: {class_names[:5]}")

    except Exception as e:
        logger.error(f"加载数据集流或提取类别名称时失败: {e}")
        logger.error("请确保你已通过 'huggingface-cli login' 登录，并同意了数据集的使用条款。")
        return

    # 处理并保存训练集 (测试样本)
    if 'train' in dataset_stream and dataset_stream['train'] is not None:
        # logger.info(f"将从 'train' 划分下载并保存 {NUM_SAMPLES_TO_TEST} 个样本进行测试。")
        logger.info(f"将从 'train' 划分下载并保存所有样本。") # 更新日志信息
        try:
            if class_names is None:
                 logger.error("类别名称未能成功加载，无法处理训练集。")
            else:
                # train_subset = dataset_stream['train'].take(NUM_SAMPLES_TO_TEST) # 移除 .take()
                train_subset = dataset_stream['train'] # 处理完整划分
                save_image_split(train_subset, TRAIN_DIR, class_names, "train")
        except Exception as e:
            logger.error(f"处理 'train' 划分的测试样本时出错: {e}")
    else:
        logger.warning("在数据集中未找到 'train' 划分或该划分无法处理。")

    # 处理并保存验证集 (测试样本)
    val_split_to_process = None
    actual_val_split_name = None 

    if 'validation' in dataset_stream and dataset_stream['validation'] is not None:
        val_split_to_process = dataset_stream['validation']
        actual_val_split_name = 'validation'
    elif 'val' in dataset_stream and dataset_stream['val'] is not None: # 有些数据集可能用 'val'
        val_split_to_process = dataset_stream['val']
        actual_val_split_name = 'val'

    if val_split_to_process:
        # logger.info(f"将从 '{actual_val_split_name}' 划分下载并保存 {NUM_SAMPLES_TO_TEST} 个样本进行测试。")
        logger.info(f"将从 '{actual_val_split_name}' 划分下载并保存所有样本。") # 更新日志信息
        try:
            if class_names is None:
                logger.error("类别名称未能成功加载，无法处理验证集。")
            else:
                # val_subset = val_split_to_process.take(NUM_SAMPLES_TO_TEST) # 移除 .take()
                val_subset = val_split_to_process # 处理完整划分
                save_image_split(val_subset, VAL_DIR, class_names, "validation") # 统一使用 "validation" 作为日志名
        except Exception as e:
            logger.error(f"处理 '{actual_val_split_name}' 划分的测试样本时出错: {e}")
    else:
        logger.warning("在数据集中未找到 'validation' 或 'val' 划分，或这些划分无法处理。")
        
    # logger.info(f"测试下载处理完成！如果成功，请检查以下目录中是否各有约 {NUM_SAMPLES_TO_TEST} 个样本：")
    logger.info(f"完整数据集下载处理完成！如果成功，请检查以下目录：") # 更新日志信息
    logger.info(f"训练数据应位于: {TRAIN_DIR.resolve()}")
    logger.info(f"验证数据应位于: {VAL_DIR.resolve()}")

if __name__ == "__main__":
    main()
