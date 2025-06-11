import os
from pathlib import Path
from datasets import load_dataset, ClassLabel, get_dataset_infos 
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

def save_image_split(dataset_split, target_dir: Path, class_names: list, split_name: str, total_examples: int = None): # <--- 添加 total_examples 参数
    """
    保存数据集的特定划分 (train/validation) 到目标目录，支持断点续传。

    Args:
        dataset_split: Hugging Face 数据集的一个划分 (e.g., dataset['train']).
        target_dir: 保存该划分的根目录 (e.g., TRAIN_DIR).
        class_names: 类别名称列表 (synset IDs, e.g., ['n01440764', ...]).
        split_name: 划分的名称，用于日志和文件名 (e.g., "train").
        total_examples: 该划分的总样本数，用于tqdm进度条 (可选).
    """
    logger.info(f"开始处理和保存 {split_name} 数据集...")
    target_dir.mkdir(parents=True, exist_ok=True)

    # num_existing_files_per_class 存储每个类别在之前运行中已保存的图片数量
    num_existing_files_per_class = {}
    logger.info(f"检查 '{target_dir}' 中已存在的 {split_name} 图片以支持断点续传...")
    for synset_id in class_names:
        class_specific_dir = target_dir / synset_id
        class_specific_dir.mkdir(parents=True, exist_ok=True) # 确保类别目录存在
        
        # 构造glob模式以匹配之前保存的文件
        # 文件名格式: f"{split_name}_{synset_id}_{count:08d}.jpg"
        try:
            existing_files = list(class_specific_dir.glob(f"{split_name}_{synset_id}_*.jpg"))
            num_existing_files_per_class[synset_id] = len(existing_files)
            if num_existing_files_per_class[synset_id] > 0:
                logger.info(f"  类别 {synset_id}: 发现 {num_existing_files_per_class[synset_id]} 张已存在的图片。将从之后继续。")
        except Exception as e:
            logger.warning(f"  检查类别 {synset_id} 的现有文件时出错: {e}. 假定为0张。")
            num_existing_files_per_class[synset_id] = 0


    # current_class_example_seen_count 追踪当前运行中，从数据流中为每个类别看到的样本总数
    current_class_example_seen_count = {synset_id: 0 for synset_id in class_names}
    
    new_files_saved_this_run = 0
    skipped_files_this_run = 0

    # 使用 total_examples 更新 tqdm 调用
    for i, example in enumerate(tqdm(dataset_split, total=total_examples, desc=f"保存 {split_name} 图片", unit="img", dynamic_ncols=True)):
        synset_id_for_example = "N/A" # 用于错误日志
        try:
            image = example['image']
            label_idx = example['label']
            
            synset_id_for_example = class_names[label_idx]
            
            current_class_example_seen_count[synset_id_for_example] += 1
            
            # 如果当前类别遇到的样本的序号小于或等于该类别已存在的样本数，则跳过
            # 这意味着这个样本在之前的运行中已经被处理过了
            if current_class_example_seen_count[synset_id_for_example] <= num_existing_files_per_class.get(synset_id_for_example, 0):
                skipped_files_this_run +=1
                # 为了避免日志过于频繁，可以考虑减少这里的日志输出或只在特定条件下输出
                if skipped_files_this_run % 10000 == 0 and skipped_files_this_run > 0:
                     logger.debug(f"已跳过 {skipped_files_this_run} 张已存在的图片...")
                continue # 跳过这个样本

            # 保存新图片
            class_specific_dir = target_dir / synset_id_for_example
            
            # 文件名中的计数器应该是当前类别遇到的总样本数 (因为我们已经跳过了等于 num_existing_files_per_class 的数量)
            filename_idx = current_class_example_seen_count[synset_id_for_example]
            filename = f"{split_name}_{synset_id_for_example}_{filename_idx:08d}.jpg"
            filepath = class_specific_dir / filename
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            image.save(filepath, format="JPEG")
            new_files_saved_this_run += 1

        except Exception as e:
            logger.error(f"处理 {split_name} 数据时发生错误 (迭代索引 {i}, 标签 {example.get('label', 'N/A')}, synset {synset_id_for_example}): {e}")
            # logger.error(f"问题数据: {example}") # 打印此项会非常冗长

    logger.info(f"{split_name} 数据集处理完成。")
    if skipped_files_this_run > 0:
        logger.info(f"  本轮跳过的已存在图片数量: {skipped_files_this_run}")
    logger.info(f"  本轮新保存的图片数量: {new_files_saved_this_run}")
    
    total_files_in_split_after_run = 0
    for synset_id_count in class_names:
        class_specific_dir_count = target_dir / synset_id_count
        try:
            total_files_in_split_after_run += len(list(class_specific_dir_count.glob(f"{split_name}_{synset_id_count}_*.jpg")))
        except Exception: # pragma: no cover
            pass # 目录可能不存在等
    logger.info(f"  '{target_dir}' 中该划分当前总图片数量: {total_files_in_split_after_run}")

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

    # --- 获取数据集总样本数以用于进度条 ---
    train_total_examples = None
    val_total_examples = None # This will store the count for 'validation' or 'val' if found
    try:
        logger.info(f"尝试获取 '{DATASET_NAME}' 数据集信息以显示进度条总数...")
        all_configs_info = get_dataset_infos(DATASET_NAME) # e.g. "imagenet-1k"
        
        # 对于 'imagenet-1k', 配置名称通常是 'imagenet-1k' 本身
        dataset_info = all_configs_info.get(DATASET_NAME) 

        if dataset_info and dataset_info.splits:
            if 'train' in dataset_info.splits:
                train_total_examples = dataset_info.splits['train'].num_examples
                if train_total_examples:
                     logger.info(f"获取到训练集总样本数: {train_total_examples}")
                else: # pragma: no cover
                     logger.warning("训练集总样本数为 None 或 0。")
            
            # 对于验证集, 检查 'validation' 然后 'val'
            if 'validation' in dataset_info.splits:
                val_total_examples = dataset_info.splits['validation'].num_examples
                if val_total_examples:
                    logger.info(f"获取到 'validation' 划分总样本数: {val_total_examples}")
                else: # pragma: no cover
                    logger.warning("'validation' 划分总样本数为 None 或 0。")
            elif 'val' in dataset_info.splits: # 'val' 作为备选
                val_total_examples = dataset_info.splits['val'].num_examples
                if val_total_examples:
                    logger.info(f"获取到 'val' 划分总样本数: {val_total_examples}")
                else: # pragma: no cover
                    logger.warning("'val' 划分总样本数为 None 或 0。")
        else: # pragma: no cover
            logger.warning(f"未能从 get_dataset_infos 获取 '{DATASET_NAME}' 的详细信息。进度条可能不显示总数。")

    except Exception as e: # pragma: no cover
        logger.warning(f"获取数据集总样本数时出错: {e}。进度条可能不显示总数。")
    # --- ---

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
                save_image_split(train_subset, TRAIN_DIR, class_names, "train", total_examples=train_total_examples) # <--- 传递 total_examples
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
                # 使用之前获取的 val_total_examples，它对应 'validation' 或 'val'
                save_image_split(val_subset, VAL_DIR, class_names, "validation", total_examples=val_total_examples) # <--- 传递 total_examples
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
