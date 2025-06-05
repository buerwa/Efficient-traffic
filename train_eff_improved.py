import torch
import torch.nn as nn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
import os
import json
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from datetime import datetime
import torchvision.utils as vutils


def main():
    # 使用GPU训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 超参数配置
    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 0.0002
    WEIGHT_DECAY = 1e-4

    # 创建带时间戳的日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs_eff/trafficnet_{timestamp}"
    writer = SummaryWriter(log_dir)

    # 数据增强和预处理
    data_transform = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomRotation(15),  # 增加旋转角度
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.1),  # 添加垂直翻转
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                ),
                transforms.RandomGrayscale(p=0.1),  # 随机灰度化
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),  # ImageNet标准化
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
    }

    # 获取数据路径
    data_root = os.path.abspath(os.path.join(os.getcwd()))
    image_path = os.path.join(data_root, "trafficnet")

    # Windows系统下设置num_workers=0避免多进程问题
    num_workers = 0 if os.name == "nt" else 4

    # 加载数据集
    train_dataset = datasets.ImageFolder(
        root=os.path.join(image_path, "train"), transform=data_transform["train"]
    )
    train_num = len(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    validate_dataset = datasets.ImageFolder(
        root=os.path.join(image_path, "val"), transform=data_transform["val"]
    )
    val_num = len(validate_dataset)

    validate_loader = torch.utils.data.DataLoader(
        validate_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # 验证集不需要打乱
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # 类别映射
    car_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in car_list.items())
    num_classes = len(cla_dict)

    # 保存类别索引
    json_str = json.dumps(cla_dict, indent=4)
    with open("class_indices.json", "w") as json_file:
        json_file.write(json_str)

    print(f"数据集信息:")
    print(f"训练集样本数: {train_num}")
    print(f"验证集样本数: {val_num}")
    print(f"类别数: {num_classes}")
    print(f"类别映射: {cla_dict}")

    # 记录数据集信息到TensorBoard
    writer.add_text(
        "Dataset/Info",
        f"""
    训练集样本数: {train_num}
    验证集样本数: {val_num}
    类别数: {num_classes}
    类别: {list(cla_dict.values())}
    """,
    )

    # 可视化数据样本
    def visualize_samples():
        """可视化训练数据样本"""
        dataiter = iter(train_loader)
        images, labels = next(dataiter)

        # 反归一化用于显示
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        images_denorm = images * std + mean
        images_denorm = torch.clamp(images_denorm, 0, 1)

        # 创建网格图像
        img_grid = vutils.make_grid(images_denorm[:8], nrow=4, normalize=False)
        writer.add_image("Dataset/Sample_Images", img_grid)

        # 记录类别分布
        class_counts = torch.bincount(labels)
        for i, count in enumerate(class_counts):
            writer.add_scalar(f"Dataset/Class_Distribution/{cla_dict[i]}", count.item())

    visualize_samples()

    # 初始化模型
    net = EfficientNet.from_pretrained("efficientnet-b1", num_classes=num_classes)
    net.to(device)

    # 记录模型结构
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    writer.add_graph(net, dummy_input)

    # 损失函数和优化器
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    # 记录超参数
    writer.add_hparams(
        {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            # "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "model": "EfficientNet-B1",
        },
        {},
    )

    # 训练指标记录
    save_path = "./backend/EfficientNet.pth"
    best_acc = 0.0
    train_losses = []
    val_accuracies = []
    learning_rates = []

    # 混淆矩阵可视化函数
    def plot_confusion_matrix(y_true, y_pred, epoch):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=list(cla_dict.values()),
            yticklabels=list(cla_dict.values()),
        )
        plt.title(f"Confusion Matrix - Epoch {epoch+1}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        # 保存图像到TensorBoard
        writer.add_figure(f"Confusion_Matrix/Epoch_{epoch+1}", plt.gcf(), epoch)
        plt.close()

    # 计算每类准确率
    def calculate_class_accuracy(y_true, y_pred):
        """计算每个类别的准确率"""
        class_correct = {}
        class_total = {}

        for i in range(num_classes):
            class_name = cla_dict[i]
            class_correct[class_name] = 0
            class_total[class_name] = 0

        for true_label, pred_label in zip(y_true, y_pred):
            class_name = cla_dict[true_label]
            class_total[class_name] += 1
            if true_label == pred_label:
                class_correct[class_name] += 1

        class_accuracies = {}
        for class_name in class_correct:
            if class_total[class_name] > 0:
                class_accuracies[class_name] = (
                    class_correct[class_name] / class_total[class_name]
                )
            else:
                class_accuracies[class_name] = 0.0

        return class_accuracies

    print("开始训练...")
    total_start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*50}")

        # ========== 训练阶段 ==========
        net.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        time_start = time.perf_counter()

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 计算训练准确率
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            # 打印训练进度
            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print(f"\rtrain loss: {int(rate * 100):3d}%[{a}->{b}]{loss:.3f}", end="")

        train_time = time.perf_counter() - time_start
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = correct_predictions / total_predictions

        print(f"\n训练时间: {train_time:.2f}s")
        print(f"训练损失: {avg_train_loss:.4f}")
        print(f"训练准确率: {train_accuracy:.4f}")

        # ========== 验证阶段 ==========
        net.eval()
        val_loss = 0.0
        correct = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for val_images, val_labels in validate_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)

                outputs = net(val_images)
                loss = loss_function(outputs, val_labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == val_labels).sum().item()

                # 收集预测结果用于混淆矩阵
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(val_labels.cpu().numpy())

        val_accuracy = correct / val_num
        avg_val_loss = val_loss / len(validate_loader)

        print(f"验证损失: {avg_val_loss:.4f}")
        print(f"验证准确率: {val_accuracy:.4f}")

        # ========== 记录到TensorBoard ==========
        # 基本指标
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_accuracy, epoch)
        writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)
        writer.add_scalar("Time/Train_Time_Per_Epoch", train_time, epoch)

        # 学习率
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Learning_Rate", current_lr, epoch)
        learning_rates.append(current_lr)

        # 每类准确率
        class_accuracies = calculate_class_accuracy(all_labels, all_predictions)
        for class_name, acc in class_accuracies.items():
            writer.add_scalar(f"Class_Accuracy/{class_name}", acc, epoch)

        # 模型权重和梯度分布
        for name, param in net.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f"Gradients/{name}", param.grad, epoch)
            writer.add_histogram(f"Weights/{name}", param, epoch)

        # 每5个epoch绘制混淆矩阵
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            plot_confusion_matrix(all_labels, all_predictions, epoch)

        # 保存最佳模型
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "class_to_idx": car_list,
                },
                save_path,
            )
            print(f"✅ 保存最佳模型! 准确率: {best_acc:.4f}")

        # 学习率调度
        scheduler.step(val_accuracy)

        # 记录历史数据
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_accuracy)

    # ========== 训练完成后的总结 ==========
    total_duration = time.perf_counter() - total_start_time
    average_epoch_duration = total_duration / EPOCHS

    print(f"\n{'='*60}")
    print("🎉 训练完成!")
    print(f"{'='*60}")
    print(f"总训练时间: {total_duration:.2f}s ({total_duration/60:.2f}分钟)")
    print(f"平均每轮训练时间: {average_epoch_duration:.2f}s")
    print(f"最佳验证准确率: {best_acc:.4f}")
    print(f"模型保存路径: {save_path}")
    print(f"TensorBoard日志: {log_dir}")

    # 最终统计图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 损失曲线
    epochs_range = range(1, EPOCHS + 1)
    ax1.plot(epochs_range, train_losses, "b-", label="Training Loss")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # 准确率曲线
    ax2.plot(epochs_range, val_accuracies, "r-", label="Validation Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    # 学习率曲线
    ax3.plot(epochs_range, learning_rates, "g-", label="Learning Rate")
    ax3.set_title("Learning Rate Schedule")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Learning Rate")
    ax3.legend()
    ax3.grid(True)
    ax3.set_yscale("log")

    # 最终类别准确率
    final_class_acc = calculate_class_accuracy(all_labels, all_predictions)
    classes = list(final_class_acc.keys())
    accuracies = list(final_class_acc.values())
    ax4.bar(classes, accuracies)
    ax4.set_title("Final Class Accuracies")
    ax4.set_xlabel("Class")
    ax4.set_ylabel("Accuracy")
    ax4.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    writer.add_figure("Training_Summary", fig)
    plt.savefig(f"{log_dir}/training_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 保存训练历史
    training_history = {
        "train_losses": train_losses,
        "val_accuracies": val_accuracies,
        "learning_rates": learning_rates,
        "best_accuracy": best_acc,
        "total_time": total_duration,
        "class_accuracies": final_class_acc,
    }

    with open(f"{log_dir}/training_history.json", "w") as f:
        json.dump(training_history, f, indent=4)

    # 生成分类报告
    print("\n📊 详细分类报告:")
    print(
        classification_report(
            all_labels, all_predictions, target_names=list(cla_dict.values())
        )
    )

    writer.close()


if __name__ == "__main__":
    main()
