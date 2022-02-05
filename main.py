from utils import Model, Encoder, Decoder,View, show_image, sampling
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import os.path
import numpy as np

import torch as pt

pt.manual_seed(2022)


def model_acc(loader):
    # mlp.eval()
    correct = 0
    total = 0
    with pt.no_grad():
        for x, y in loader:
            x = x.view(batch_size, x_dim)

            _, mu, log_var = model(x)
            latent = sampling(mu, pt.exp(0.5 * log_var), samples)
            outputs = mlp(latent)

            _, predicted = pt.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y.to(DEVICE)).sum().item()
        if loader == train_loader:
            print(f"Accuracy on train data: {100 * correct // total} %")
        else:
            print(f"Accuracy on test data: {100 * correct // total} %")


dataset_path = "~/datasets"

cuda = True
DEVICE = pt.device("cuda" if cuda else "cpu")


batch_size = 1
x_dim = 784
hidden_dim = 400
latent_dim = 2
lr = 0.001
epochs = 1000
samples = 10


mnist_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

kwargs = {"num_workers": 1, "pin_memory": True}

train_dataset = MNIST(
    dataset_path, transform=mnist_transform, train=True, download=True
)
test_dataset = MNIST(
    dataset_path, transform=mnist_transform, train=False, download=True
)

# randomly select 100 samples from the dataset 
# Please comment out below two lines if you want to train the model with full mnist
# train_dataset.data = train_dataset.data[pt.randperm(len(train_dataset))[0:100]]
# test_dataset.data = test_dataset.data[pt.randperm(len(test_dataset))[0:50]]

train_loader = pt.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs
)
test_loader = pt.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False, **kwargs
)


# Initializing models
encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(
    latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim
)

model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)

# creating a mlp model
mlp = pt.nn.Sequential(
    View((1, latent_dim * samples)),
    pt.nn.Linear(latent_dim * samples, (latent_dim * samples) * 2),
    pt.nn.LayerNorm((latent_dim * samples) * 2),
    pt.nn.ReLU(),
    pt.nn.Dropout(0.0),
    pt.nn.Linear((latent_dim * samples) * 2, (latent_dim * samples) * 2),
    pt.nn.LayerNorm((latent_dim * samples) * 2),
    pt.nn.ReLU(),
    pt.nn.Dropout(0.0),
    pt.nn.Linear((latent_dim * samples) * 2, (latent_dim * samples) * 2),
    pt.nn.LayerNorm((latent_dim * samples) * 2),
    pt.nn.ReLU(),
    pt.nn.Dropout(0.0),
    pt.nn.Linear((latent_dim * samples) * 2, (latent_dim * samples) * 2),
    pt.nn.LayerNorm((latent_dim * samples) * 2),
    pt.nn.ReLU(),
    pt.nn.Dropout(0.0),
    pt.nn.Linear((latent_dim * samples) * 2, 10),
)

print(mlp)


# Load model
model.load_state_dict(pt.load("model.pth", map_location=DEVICE))


for param in model.parameters():
    param.requires_grad = False


if os.path.exists("mlp.pth"):
    mlp.load_state_dict(pt.load("mlp.pth", map_location=DEVICE))


# optimizer = pt.optim.Adam(mlp.parameters(), lr=lr)
optimizer = pt.optim.SGD(mlp.parameters(), lr=lr)
loss_function = pt.nn.CrossEntropyLoss()

model.eval()


mlp.train()


losses = []
for e in range(epochs):
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.view(batch_size, x_dim)
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        with pt.no_grad():
            _, mu, log_var = model(x)

        optimizer.zero_grad()
        # sampling n latents
        latent = sampling(mu, pt.exp(0.5 * log_var), samples)
        output = mlp(latent).to(DEVICE)

        loss = loss_function(output, y)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()

    if e % 50 == 0:
        print("epoch: ", str(e), "Avg Loss: ", pt.tensor(losses).mean())
        pt.save(mlp.state_dict(), "mlp.pth")
        # Prediction
        model_acc(train_loader)
        model_acc(test_loader)
