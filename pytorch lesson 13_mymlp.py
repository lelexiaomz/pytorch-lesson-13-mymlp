# ============================================================
# PyTorch 第十三课：MLP 回归 / PyTorch Lesson 13: MLP Regression
#
# Author: lelexiaomz
#
# 目的：训练一个多层感知机（MLP），拟合非线性函数 y = x²。
# Purpose: Train a multilayer perceptron (MLP) to approximate the nonlinear function y = x².
#
# 程序结构 / Main components:
# ├── MyDataset：生成 x 和 y = x² 样本 / Generates x and y = x² samples
# ├── MyMLP：定义回归模型 / Defines the regression model
# ├── 数据集划分、标准化和 DataLoader / Dataset splitting, normalization, and DataLoaders
# ├── 模型训练、验证、早停和测试 / Training, validation, early stopping, and testing
# ├── 新数据预测 / Prediction on new data
# └── Loss 曲线绘制 / Loss-curve plotting
# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset

import copy
import matplotlib.pyplot as plt



# ============================================================
# 1. 创建数据集 / Create the dataset
# ============================================================

class MyDataset(Dataset):
    """提供用于拟合 y = x² 的输入和目标样本。
    Dataset of input and target samples for fitting y = x².
    """

    def __init__(self):
        # 在 [-19, 30] 区间均匀生成 500 个 x 值，并增加特征维度。
        # Generate 500 evenly spaced x values in [-19, 30] and add a feature dimension.
        self.x = torch.linspace(-19, 30, 500).unsqueeze(1)  # Shape: [500, 1]
        # 计算目标值 y = x²。
        # Compute the target values y = x².
        self.y = self.x ** 2  # Shape: [500, 1]

    def __len__(self):
        """返回数据集中的样本数。 / Return the number of samples in the dataset."""
        return len(self.x)

    def __getitem__(self, index):
        """返回指定索引处的 (x, y) 样本。 / Return the (x, y) sample at the given index."""
        return self.x[index], self.y[index]



# ============================================================
# 2. 创建模型 / Create the model
# ============================================================

class MyMLP(nn.Module):
    """用于一维回归的 MLP，网络结构为 1 → 64 → 1。
    MLP for one-dimensional regression with architecture 1 → 64 → 1.
    """

    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(1,64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64,1)

    def forward(self, x):

        # 依次执行线性变换、ReLU 激活和输出线性变换。
        # Apply a linear transformation, ReLU activation, and the output linear transformation in sequence.
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x


# ============================================================
# 3. 创建并划分数据集 / Create and split the dataset
# ============================================================

def prepare_dataset():
    """生成数据并随机划分为训练集、验证集和测试集。
    Generate the data and randomly split it into training, validation, and test sets.
    """

    # 创建完整数据集 / Create the complete dataset

    dataset = MyDataset()
    # 获取数据集中的样本总数 / Get the total number of samples in the dataset
    total_num_dataset = len(dataset)

    # 打印数据集大小 / Print the dataset size
    print('========= Total number of Dataset =========')
    print(f'Total Number of samples: {total_num_dataset}')



    # 按 70% / 15% / 15% 划分训练集、验证集和测试集。
    # Split the data into 70% training, 15% validation, and 15% test sets.

    # 训练集占总样本数的 70%。 / Use 70% of the samples for training.
    train_size = int(0.7 * total_num_dataset)

    # 验证集占总样本数的 15%。 / Use 15% of the samples for validation.
    val_size = int(0.15 * total_num_dataset)

    # 将剩余样本分配给测试集，处理整数取整造成的差异。
    # Assign the remaining samples to the test set to account for integer rounding.
    test_size = total_num_dataset - train_size - val_size

    # 随机划分数据集；各子集互不重叠。
    # Randomly split the dataset into non-overlapping subsets.
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


    # 打印训练集大小 / Print the training-set size
    print('\n========= Train dataset =========')
    print(f'Train dataset size:      {train_size}') 
    # 可取消注释以查看部分训练样本。
    # Uncomment to inspect a few training samples.
    # for i in range(10):
    #     x, y = train_dataset[i]
    #     print(f'Train sample {i}: x = {x.item()}, y = {y.item()}')

    # 打印验证集大小 / Print the validation-set size
    print('\n========= Validation dataset =========')
    print(f'Validation dataset size:  {val_size}')
    # 可取消注释以查看部分验证样本。
    # Uncomment to inspect a few validation samples.
    # for i in range(10):
    #     x, y = val_dataset[i]
    #     print(f'Val sample {i}: x = {x.item()}, y = {y.item()}')

    # 打印测试集大小 / Print the test-set size
    print('\n========= Test dataset =========')
    print(f'Test dataset size:        {test_size}')
    # 可取消注释以查看部分测试样本。
    # Uncomment to inspect a few test samples.
    # for i in range(10):
    #     x, y = test_dataset[i]
    #     print(f'Test sample {i}: x = {x.item()}, y = {y.item()}')

    return train_dataset, val_dataset, test_dataset



# ============================================================
# 4. 计算标准化统计量 / Calculate normalization statistics
# ============================================================

def calculate_normalization_stats(train_dataset):
    """根据训练集计算输入和目标的均值与标准差。
    Calculate input and target means and standard deviations from the training set.
    """

    # 分别收集训练集中的输入 x 和目标 y，并沿新维度堆叠为张量。
    # Collect training inputs x and targets y, stacking samples along a new dimension.
    train_x = torch.stack([x for x, y in train_dataset])
    train_y = torch.stack([y for x, y in train_dataset])


    # 计算训练输入和目标各自的均值与标准差。
    # Calculate the mean and standard deviation of the training inputs and targets.
    train_x_mean = train_x.mean()

    train_x_std = train_x.std()

    train_y_mean = train_y.mean()

    train_y_std = train_y.std()

    print('\n============== Mean and Std ==============')
    print(f'train_x_mean = {train_x_mean}')
    print(f'train_x_std  = {train_x_std}')

    print(f'train_y_mean = {train_y_mean}')
    print(f'train_y_std  = {train_y_std}')

    return train_x_mean, train_x_std, train_y_mean, train_y_std


# ============================================================
# 5. 标准化数据集 / Normalize datasets
# ============================================================

def normalization_dataset(dataset, train_x_mean, train_x_std, train_y_mean, train_y_std):
    """使用训练集统计量标准化一个数据集，并返回输入和目标张量。
    Normalize a dataset with training-set statistics and return input and target tensors.
    """

    # 使用训练集统计量标准化任意数据集，避免使用验证集或测试集信息计算统计量。
    # Normalize a dataset with training-set statistics to avoid deriving statistics from validation or test data.
    x_norm_list = []
    y_norm_list = []

    for x, y in dataset:
        # 使用 z-score 公式分别标准化输入和目标。
        # Apply z-score normalization to the input and target.
        x_norm = (x - train_x_mean) / train_x_std
        y_norm = (y - train_y_mean) / train_y_std

        # 保存当前样本的标准化结果 / Store the normalized sample
        x_norm_list.append(x_norm)
        y_norm_list.append(y_norm)

    # 将样本列表堆叠为张量 / Stack the sample lists into tensors
    normalized_x = torch.stack(x_norm_list)
    normalized_y = torch.stack(y_norm_list)

    return normalized_x , normalized_y


# ============================================================
# 6. 创建 DataLoader / Create data loaders
# ============================================================

def create_data_loaders(
        train_x_norm,
        train_y_norm,
        val_x_norm,
        val_y_norm,
        test_x_norm,
        test_y_norm,
        train_batch_size,
        val_batch_size,
        test_batch_size
):
    """为三个数据集创建 DataLoader，并分别使用指定的批次大小。
    Create DataLoaders for the three datasets using their specified batch sizes.
    """

    # 将标准化后的输入和目标配对成 TensorDataset。
    # Pair normalized inputs and targets in TensorDataset objects.
    train_dataset_norm = TensorDataset(train_x_norm, train_y_norm)

    val_dataset_norm = TensorDataset(val_x_norm, val_y_norm)

    test_dataset_norm = TensorDataset(test_x_norm, test_y_norm)


    # 为每个数据划分创建 DataLoader；仅训练集打乱样本顺序。
    # Create a DataLoader for each split; shuffle samples only in the training loader.

    # 训练数据加载器 / Training DataLoader
    train_loader = DataLoader(train_dataset_norm, batch_size= train_batch_size, shuffle=True)

    # 验证数据加载器 / Validation DataLoader
    val_loader = DataLoader(val_dataset_norm, batch_size= val_batch_size, shuffle=False)

    # 测试数据加载器 / Test DataLoader
    test_loader = DataLoader(test_dataset_norm, batch_size= test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ============================================================
# 7. 创建模型、损失函数和优化器 / Create model components
# ============================================================

def create_model_components(learning_rate):
    """创建 MLP、MSE 损失函数、Adam 优化器和 StepLR 调度器。
    Create the MLP, MSE loss function, Adam optimizer, and StepLR scheduler.
    """

    # 创建 MLP 回归模型 / Create the MLP regression model

    # 以下 Sequential 写法与 MyMLP 等价，可用于对照理解网络结构。
    # The Sequential version below is equivalent to MyMLP and can be used to compare the architecture.
    # model = nn.Sequential(
    #     nn.Linear(1, 64),  # 输入层 / Input layer
    #     nn.ReLU(),         # 激活函数 / Activation function
    #     nn.Linear(64, 1)   # 输出层 / Output layer
    # )

    model = MyMLP()

    print('\n============== Model Architecture ==============')
    print(model)

    # 可取消注释以打印模型的所有可训练参数名称和形状。
    # Uncomment to print the names and shapes of all trainable model parameters.
    #print(list(model.named_parameters()))

    for name, param in model.named_parameters():
        print(name,param.size())


    # 使用均方误差（MSE）衡量预测值与目标值之间的差异。
    # Use mean squared error (MSE) to measure the difference between predictions and targets.
    loss_function = nn.MSELoss()



    # 使用 Adam 优化器更新模型参数 / Use the Adam optimizer to update model parameters
    optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

    # 本程序每个 Epoch 调用一次 scheduler.step()，因此每 step_size 个 Epoch 将学习率乘以 gamma。
    # This script calls scheduler.step() once per epoch, so the learning rate is multiplied by gamma every step_size epochs.
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    return model, loss_function, optimizer, scheduler


# ============================================================
# 8. 训练一个 Epoch / Train for one epoch
# ============================================================

def train_one_epoch(model, data_loader, loss_function, optimizer):
    """训练模型一个 Epoch，并返回按样本数加权的平均损失。
    Train the model for one epoch and return the sample-weighted mean loss.
    """

    # 启用训练模式 / Enable training mode
    model.train()

    # 累计按样本加权的损失和样本数 / Accumulate sample-weighted loss and the number of samples
    total_train_loss = 0.0
    total_train_samples = 0

    for x, y in data_loader:

        # 1. 前向传播，计算预测值 / Run the forward pass to compute predictions
        y_pred = model(x)

        # 2. 计算当前批次的 MSE 损失 / Compute the MSE loss for this batch
        loss = loss_function(y_pred, y)

        # 3. 清除上一批次累积的梯度 / Clear gradients accumulated from the previous batch
        optimizer.zero_grad()

        # 4. 反向传播，计算梯度 / Backpropagate to compute gradients
        loss.backward()

        # 5. 更新模型参数 / Update model parameters
        optimizer.step()

        batch_size = x.size(0)
        # loss 是批次平均值，因此乘以批次样本数后再累计。
        # loss is the batch mean, so multiply it by the batch size before accumulating.
        total_train_loss += loss.item() * batch_size
        # 累计样本数，用于最后计算全数据集的平均损失。
        # Count samples to compute the mean loss over the full dataset.
        total_train_samples += batch_size

    # 计算整个数据集按样本加权的平均训练损失。
    # Compute the sample-weighted mean training loss over the dataset.
    avg_train_loss = total_train_loss / total_train_samples

    return avg_train_loss



# ============================================================
# 9. 验证模型 / Validate the model
# ============================================================

def validate(model, data_loader, loss_function):
    """在验证集上计算平均损失，不更新模型参数。
    Compute the mean loss on the validation set without updating model parameters.
    """

    # 切换到评估模式 / Switch to evaluation mode
    model.eval()

    # 累计按样本加权的损失和样本数 / Accumulate sample-weighted loss and the number of samples
    total_val_loss = 0.0
    total_val_samples = 0

    # 验证阶段不需要计算梯度 / Gradients are not needed during validation
    with torch.no_grad():

        for x, y in data_loader:

            # 前向传播，计算预测值 / Run the forward pass to compute predictions
            y_pred = model(x)

            # 计算当前批次的 MSE 损失 / Compute the MSE loss for this batch
            loss = loss_function(y_pred, y)

            batch_size = x.size(0)
            # 按样本数还原并累计批次平均损失。
            # Scale the batch-mean loss by the batch size before accumulating.
            total_val_loss += loss.item() * batch_size
            # 累计样本数 / Count samples
            total_val_samples += batch_size

    # 计算整个验证集按样本加权的平均损失。
    # Compute the sample-weighted mean loss over the full validation set.
    avg_val_loss = total_val_loss / total_val_samples

    return avg_val_loss





# ============================================================
# 10. 完整训练流程 / Full training procedure
# ============================================================

def train_model(model, train_loader, val_loader, loss_function, optimizer, scheduler, epochs, patience):
    """训练模型，按验证损失执行早停，并恢复最佳模型状态。
    Train the model, apply early stopping based on validation loss, and restore the best model state.

    Returns:
        最佳模型状态、最佳验证损失、最佳 Epoch，以及训练和验证损失历史。
        Best model state, best validation loss, best epoch, and training/validation loss histories.
    """

    # 保存每个 Epoch 的训练损失和验证损失 / Store training and validation losses for each epoch
    train_loss = []
    val_loss = []


    # 早停所需状态 / State used by early stopping

    # 初始化为无穷大，使第一个验证损失必然成为当前最佳值。
    # Initialize to infinity so the first validation loss becomes the current best.
    best_val_loss = float('inf')

    # 保存验证损失最低时的模型状态字典副本。
    # Keep a copy of the model state with the lowest validation loss.
    best_model_state = None

    # 最佳模型检查点的文件路径，相对于当前工作目录。
    # File path for the best-model checkpoint, relative to the current working directory.
    best_model_path = 'lesson13_best_model.pth'

    # 记录验证损失连续未改善的 Epoch 数。
    # Count consecutive epochs without an improvement in validation loss.
    patience_counter = 0

    # 记录验证损失最低的 Epoch 编号（从 1 开始）。
    # Store the 1-based epoch number with the lowest validation loss.
    best_epoch = 0

    for epoch in range(epochs):

        # 训练一个 Epoch / Train for one epoch
        avg_train_loss = train_one_epoch(model,train_loader, loss_function, optimizer)

        # 在验证集上评估模型 / Evaluate on the validation set
        avg_val_loss = validate(model, val_loader, loss_function)

        # 保存本轮损失 / Store this epoch's losses
        train_loss.append(avg_train_loss)
        val_loss.append(avg_val_loss)

        # 完成当前 Epoch 后，根据 StepLR 的设置更新学习率。
        # Update the learning rate after this epoch according to the StepLR schedule.
        scheduler.step()


        # 检查验证损失是否低于历史最佳值 / Check whether validation loss beats the best so far
        
        # 验证损失改善时，更新最佳记录并重置早停计数器。
        # When validation loss improves, update the best record and reset the early-stopping counter.
        if avg_val_loss < best_val_loss:  

            # 更新最低验证损失 / Update the best validation loss
            best_val_loss = avg_val_loss

            # 保存当前最佳 Epoch 编号 / Store the current best epoch number
            best_epoch = epoch + 1

            # 重置未改善计数 / Reset the no-improvement counter
            patience_counter = 0

            # 深拷贝当前参数，并将检查点写入文件。
            # Deep-copy the current parameters and write the checkpoint to disk.
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, best_model_path)
            print('Model state saved.')

            print(f'\n ================ Validation mode, epoch: {epoch} ================')
            print(f'Validation Loss improved: {best_val_loss}')
            print(f'Avg_train_loss:           {avg_train_loss}') 

            current_lr = optimizer.param_groups[0]['lr']
            print(f'Current learning rate:    {current_lr}') 
            
            # 可取消注释以在 train_one_epoch() 中打印一个批次的目标值和预测值。
            # Uncomment the diagnostic print in train_one_epoch() to inspect a batch's targets and predictions.
        else:
            # 验证损失没有改善，增加早停计数器。
            # Validation loss did not improve; increment the early-stopping counter.
            patience_counter += 1


        # 连续 patience 个 Epoch 未改善时提前停止 / Stop if there is no improvement for patience epochs

        if patience_counter >= patience:

            # 当未改善计数达到 patience 时，提前结束训练。
            # Stop training early when the no-improvement counter reaches patience.
            print(f'\n==================== Early stopping at epoch NO.{epoch + 1} ====================\n')
            print('------ Hyperparameters ------')
            print(f'manual_seed(42)  = True')
            print(f'Train size =       {len(train_loader.dataset)}')
            print(f'Val size   =       {len(val_loader.dataset)}')
            print(f'Train Batch size = {train_loader.batch_size}')
            print(f'Val   Batch size = {val_loader.batch_size}')
            print(f'Epoch            = {epochs}')
            print(f'Initial learning rate = {optimizer.defaults["lr"]}')
            print(f'Current learning rate = {optimizer.param_groups[0]["lr"]}')
            print(f'Scheduler             = StepLR(step_size={scheduler.step_size}, gamma={scheduler.gamma})')
            print(f'Patience counter = {patience_counter}')  


            # 打印验证损失最低的 Epoch 和对应损失。
            # Print the epoch with the lowest validation loss and its loss value.
            print('\n------ Train and Validation Result ------')
            print(f'Best Val Epoch:       {best_epoch}')
            print(f'Best Validation Loss: {best_val_loss}')
            
            # 找到记录到的最低训练损失 / Find the lowest recorded training loss
            best_train_loss = min(train_loss)
            # 找到最低训练损失对应的 Epoch 编号（从 1 开始）。
            # Find the 1-based epoch number corresponding to the lowest training loss.
            best_epoch_train = train_loss.index(best_train_loss) + 1
            
            # 打印最低训练损失及其 Epoch / Print the lowest training loss and its epoch
            print(f'Best Train Epoch:     {best_epoch_train}')
            print(f'Best Train Loss:      {best_train_loss}')
                  
            print('\n================================================================')
            break

    # 仅当循环自然完成全部 Epoch（未因早停退出）时打印训练完成信息。
    # Print completion information only if the loop finishes all epochs without early stopping.
    if epoch == epochs - 1:

        print(f'\n==================== Training completed at epoch NO.{epoch + 1} ====================')
        print('------ Hyperparameters ------')
        print(f'manual_seed(42)  = True')
        print(f'Train size =       {len(train_loader.dataset)}')
        print(f'Val size   =       {len(val_loader.dataset)}')
        print(f'Train Batch size = {train_loader.batch_size}')
        print(f'Val   Batch size = {val_loader.batch_size}')
        print(f'Epoch            = {epochs}')
        print(f'Initial learning rate = {optimizer.defaults["lr"]}')
        print(f'Current learning rate = {optimizer.param_groups[0]["lr"]}')
        print(f'Scheduler             = StepLR(step_size={scheduler.step_size}, gamma={scheduler.gamma})')
        print(f'Patience counter = {patience_counter}')  
    
        # 打印验证损失最低的 Epoch 和对应损失。
        # Print the epoch with the lowest validation loss and its loss value.
        print('\n------ Train and Validation Result ------')
        print(f'Best Val Epoch:       {best_epoch}')
        print(f'Best Validation Loss: {best_val_loss}')

        # 找到记录到的最低训练损失 / Find the lowest recorded training loss
        best_train_loss = min(train_loss)
        # 找到最低训练损失对应的 Epoch 编号（从 1 开始）。
        # Find the 1-based epoch number corresponding to the lowest training loss.
        best_epoch_train = train_loss.index(best_train_loss) + 1

        # 打印最低训练损失及其 Epoch / Print the lowest training loss and its epoch
        print(f'Best Train Epoch:     {best_epoch_train}')
        print(f'Best Train Loss:      {best_train_loss}')
  
        #print('\n================================================================')

    # =====================================================
    # 恢复最佳模型 / Restore the best model
    # =====================================================
    # 将当前模型恢复为验证损失最低时保存的参数。
    # Restore the model parameters saved at the lowest validation loss.
    print('\n================ Restoring Best Model State ====================')
    print(f'Best Val Epoch: {best_epoch}')
    print(f'Best Val Loss:  {best_val_loss}')
    model.load_state_dict(best_model_state)

    return best_model_state, best_val_loss, best_epoch, train_loss, val_loss



# =========================================================
# 11. 测试模型 / Evaluate on the test set
# =========================================================

def test_model(model, test_loader, loss_function):
    """计算并打印测试集上的平均损失。 / Compute and print the mean test loss."""

    # 切换到评估模式 / Switch to evaluation mode
    model.eval()

    # 累计按样本加权的测试损失 / Accumulate sample-weighted test loss
    total_test_loss = 0.0
    
    # 统计测试样本总数 / Count the total number of test samples
    total_test_samples = 0
    

    # 测试阶段不需要计算梯度 / Gradients are not needed during testing
    with torch.no_grad():

        for x, y in test_loader:

            # 前向传播，计算预测值 / Run the forward pass to compute predictions
            y_pred = model(x)

            # 计算当前批次的 MSE 损失 / Compute the MSE loss for this batch
            loss = loss_function(y_pred, y)

            batch_size = x.size(0)

            # 按样本数还原并累计批次平均损失。
            # Scale the batch-mean loss by the batch size before accumulating.
            total_test_loss += loss.item() * batch_size

            # 累计测试样本数 / Count test samples
            total_test_samples += batch_size


    # 计算整个测试集按样本加权的平均损失。
    # Compute the sample-weighted mean loss over the full test set.
    avg_test_loss = total_test_loss / total_test_samples

    print('\n================ Test Reslut ====================')
    print(f'Test Size =       {len(test_loader.dataset)}')
    print(f'Test Batch Size = {test_loader.batch_size} ')
    print(f'Avg Test Loss =   {avg_test_loss} ')

    return avg_test_loss



# ============================================================
# 12. 预测新数据 / Predict on new data
# ============================================================

def predict(
    model,
    x,
    train_x_mean,
    train_x_std,
    train_y_mean,
    train_y_std):
    """标准化输入、预测目标，并将预测结果还原到原始尺度。
    Normalize the input, predict the target, and convert the prediction back to the original scale.
    """

    # 切换到评估模式 / Switch to evaluation mode
    model.eval()

    # 使用训练集的均值和标准差标准化输入。
    # Normalize the input using the training-set mean and standard deviation.
    x_norm = (x - train_x_mean) / train_x_std

    with torch.no_grad():

        y_pred_norm = model(x_norm)

    # 将模型输出从标准化尺度还原到原始目标尺度。
    # Convert the model output from normalized scale back to the original target scale.
    y_pred = y_pred_norm * train_y_std + train_y_mean

    return y_pred


# =========================================================
# 13. 绘制损失曲线 / Plot the loss curves
# =========================================================

def plot_loss_curve(train_loss, val_loss):
    """绘制并保存训练损失和验证损失曲线。
    Plot and save the training and validation loss curves.
    """

    # 创建新的图形窗口 / Create a new figure
    plt.figure()

    # 绘制训练损失曲线 / Plot the training loss curve
    plt.plot(train_loss, label = 'Train Loss')
    # 绘制验证损失曲线 / Plot the validation loss curve
    plt.plot(val_loss, label= 'Validation Loss')

    # 设置横坐标名称：Epoch / Set the x-axis label: Epoch
    plt.xlabel('Epoch')
    # 设置纵坐标名称：Loss / Set the y-axis label: Loss
    plt.ylabel('Loss')

    # 设置图表标题 / Set the chart title
    plt.title('Training Loss vs Validation Loss')

    # 显示两条曲线的图例 / Show the legend for both curves
    plt.legend()

    # 显示网格，便于观察损失变化 / Show a grid to make loss changes easier to observe
    plt.grid()

    # 将图形保存为 loss_curve.png / Save the figure as loss_curve.png
    plt.savefig('loss_curve.png')

    # 显示图形窗口 / Display the figure window
    plt.show()


# ============================================================
# 14. 主程序 / Main program
# ============================================================

def main():
   
    
    # 设置随机种子，保证每次运行结果相同 / Set the random seed so each run produces the same result
    torch.manual_seed(42)

    # 设置最大训练轮数 / Set the maximum number of training epochs
    epochs = 5000

    # 设置早停的 patience 值 / Set the patience for early stopping
    patience = 500

    # 分别设置训练、验证和测试批次大小 / Set batch sizes for training, validation, and testing
    train_batch_size = 32
    val_batch_size = 75
    test_batch_size =75

    # 设置 Adam 优化器的初始学习率 / Set the initial learning rate for Adam
    learning_rate = 0.01

    # 创建原始数据集，并划分为训练、验证和测试子集。
    # Create the raw dataset and split it into training, validation, and test subsets.
    train_dataset, val_dataset, test_dataset = prepare_dataset()

    # 仅根据训练集计算标准化统计量，再将同一组统计量用于所有数据集。
    # Compute normalization statistics from the training set only, then reuse them for every split.
    train_x_mean, train_x_std, train_y_mean, train_y_std = calculate_normalization_stats(train_dataset)
   

    # 使用训练集统计量标准化训练集、验证集和测试集。
    # Normalize the training, validation, and test sets using training-set statistics.

    # 标准化训练集 / Normalize the training set
    train_x_norm, train_y_norm = normalization_dataset(train_dataset, train_x_mean, train_x_std, train_y_mean, train_y_std)

    # 使用训练集统计量标准化验证集 / Normalize the validation set with training statistics
    val_x_norm, val_y_norm = normalization_dataset(val_dataset, train_x_mean, train_x_std, train_y_mean, train_y_std)

    # 使用训练集统计量标准化测试集 / Normalize the test set with training statistics
    test_x_norm, test_y_norm = normalization_dataset(test_dataset, train_x_mean, train_x_std, train_y_mean, train_y_std)

    # 创建训练、验证和测试数据加载器 / Create training, validation, and test DataLoaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_x_norm=train_x_norm, 
        train_y_norm=train_y_norm,
        val_x_norm=val_x_norm,
        val_y_norm=val_y_norm,
        test_x_norm=test_x_norm,
        test_y_norm=test_y_norm,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        test_batch_size=test_batch_size
    )


    # 创建模型、损失函数、优化器和学习率调度器。
    # Create the model, loss function, optimizer, and learning-rate scheduler.
    model, loss_function, optimizer, scheduler = create_model_components(learning_rate=learning_rate)

    # 训练模型，并在训练结束后恢复验证损失最低时的参数。
    # Train the model and restore the parameters from the epoch with the lowest validation loss.
    best_model_state, best_val_loss, best_epoch, train_loss, val_loss = train_model(
    model=model, 
    train_loader=train_loader,
    val_loader= val_loader,
    loss_function= loss_function,
    optimizer= optimizer,
    scheduler= scheduler,
    epochs= epochs,
    patience= patience
    )


    # 使用测试集评估恢复后的最佳模型 / Evaluate the restored best model on the test set
    test_loss = test_model(
        model=model,
        test_loader=test_loader,
        loss_function=loss_function
    )


    # 创建一个新输入，用于预测对应的 y = x² 值。
    # Create a new input and predict its corresponding y = x² value.
    x_new = torch.tensor([[90.0]])

    # 预测新输入对应的输出 / Predict the output for the new input
    y_pred = predict(model=model, 
        x=x_new, 
        train_x_mean=train_x_mean, 
        train_x_std=train_x_std,
        train_y_mean=train_y_mean,
        train_y_std=train_y_std)

    # 打印预测结果和相对误差 / Print the prediction and relative error
    print('\n================ Prediction Result ====================')
    print(f'x = {x_new.item()}')
    print(f'y_pred = {y_pred.item()}')
    print(f'Error = {abs(y_pred.item() - x_new.item() ** 2) / x_new.item() ** 2 :.2%}')


    # 绘制训练和验证损失曲线 / Plot the training and validation loss curves
    plot_loss_curve(train_loss=train_loss, val_loss=val_loss)




if __name__ == '__main__':
    main()

