# filepath: test_wandb.py
import wandb
import random

try:
    # 1. 初始化 wandb run
    #    - project: 你的项目名称 (如果不存在会自动创建)
    #    - entity: 你的 wandb 用户名或团队名 (可选, 如果未指定会使用默认)
    run = wandb.init(project="my-test-project", job_type="test-run")
    print(f"W&B Run initialized: {run.name} (ID: {run.id})")
    print(f"View run at: {run.url}")

    # 2. 记录一些配置信息 (可选)
    wandb.config.learning_rate = 0.01
    wandb.config.epochs = 5

    # 3. 模拟训练并记录指标
    print("Simulating training and logging metrics...")
    for epoch in range(wandb.config.epochs):
        loss = random.uniform(0, 1)
        accuracy = 1 - loss + random.uniform(-0.1, 0.1)
        accuracy = max(0, min(1, accuracy)) # 确保准确率在 0-1 之间

        # 记录指标
        wandb.log({"epoch": epoch, "loss": loss, "accuracy": accuracy})
        print(f"Epoch {epoch+1}/{wandb.config.epochs}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")

    # 4. 结束 run
    run.finish()
    print("W&B Run finished successfully.")

except Exception as e:
    print(f"An error occurred with W&B: {e}")
    if "API key" in str(e) or "authentication" in str(e):
        print("This might be an authentication issue. Please ensure you are logged in via 'wandb login'.")
