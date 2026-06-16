import argparse
import torch
from utils import accuracy, preprocess_data
from model import Model
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='C:/Users/15943/Desktop/skeleton', help='Directory of dataset.')
parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda:0')
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=4e-4, help='weight decay for optimizer')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
parser.add_argument('--batch', type=int, default=64, help='Batch size.')
args = parser.parse_args()

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
nclass, train_label, test_label, train_data, test_data = preprocess_data(args.dataset_dir)

train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

model = Model().to(args.device)

# 损失函数
criterion = torch.nn.CrossEntropyLoss().to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# 训练数据集
total_step = len(train_loader)

for epoch in tqdm(range(args.epochs)):
    # 训练网络模型
    model.train()
    for i, (data, labels) in enumerate(train_loader):
        data = data.float().to(args.device)
        labels = labels.long().to(args.device)
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        acc = accuracy(outputs, labels)
        with open('./log.txt', 'a') as f:
                f.write("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} Accuracy: {:.2f}%\n"
                .format(epoch+1, args.epochs, i+1, total_step, loss.item(), acc * 100))
        # print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f} Accuracy: {:.2f}%"
        #         .format(epoch+1, args.epochs, i+1, total_step, loss.item(), acc * 100))


    # 测试网络模型
    model.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.float().to(args.device)
            labels = labels.long().to(args.device)
            outputs = model(data)
            acc = accuracy(outputs, labels)

        with open('./log.txt', 'a') as f:
            f.write('Accuracy of the model on the test images: {:.2f} %\n'.format(100 * acc))
        # print('Accuracy of the model on the test images: {:.2f} %'.format(100 * acc))

    # 将模型保存
    torch.save(model.state_dict(), './save_models/resnet_epoch{}.ckpt'.format(epoch + 1))