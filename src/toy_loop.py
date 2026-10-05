import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=2000, noise=0.2, random_state=0)
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=0)
tr = DataLoader(
    TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr)),
    batch_size=32, shuffle=True)
Xva_t, yva_t = torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva)

model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

train_losses = []
val_accs = []

for epoch in range(20):
    model.train()
    for xb, yb in tr:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        acc = (model(Xva_t).argmax(1) == yva_t).float().mean().item()

    train_losses.append(loss.item())   # last batch's loss this epoch
    val_accs.append(acc)

    print(f"epoch {epoch+1}: last batch loss {loss.item():.3f}, val acc {acc:.3f}")

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()

ax1.plot(train_losses, color="tab:blue", label="train loss")
ax1.set_xlabel("epoch")
ax1.set_ylabel("loss", color="tab:blue")

ax2 = ax1.twinx()
ax2.plot(val_accs, color="tab:orange", label="val accuracy")
ax2.set_ylabel("accuracy", color="tab:orange")

plt.title("Toy loop: loss and accuracy per epoch")
fig.tight_layout()
plt.savefig("toy_loop_plot.png")
print("Saved toy_loop_plot.png")