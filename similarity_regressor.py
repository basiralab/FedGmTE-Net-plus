import torch
import torch.nn as nn
from scipy.stats import pearsonr

class SimilarityRegressor(nn.Module):
    def __init__(self, input_dim):
        super(SimilarityRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

def train_similarity_regressor(reg, opt, comparison_vectors, ground_truth, mask, device, epochs=2000):
    criterion = torch.nn.L1Loss().to(device)

    gts = []
    distances = []
    for i in range(mask.shape[0]):
        for j in range(i, mask.shape[0]):
            for t in range(1, mask.shape[1]):
                if mask[i, t] == 1 and mask[j, t] == 1:
                    distance = torch.abs(comparison_vectors[t-1][i] - comparison_vectors[t-1][j])
                    pc, _ = pearsonr(ground_truth[t][i].cpu(), ground_truth[t][j].cpu())
                    distances.append(distance)
                    gts.append([max(0,pc)])
    distances = torch.stack(distances, dim=0).to(device)
    gts = torch.tensor(gts, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        opt.zero_grad()
        preds = reg(distances)
        loss = criterion(gts, preds)
        loss.backward(retain_graph=True)
        opt.step()

        # Print loss for monitoring
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}: Loss - {loss.item()}")

def similarity_imputation(reg, comparison_vectors, ground_truth, mask, device):
    pcs = torch.zeros(mask.shape[0], mask.shape[0], mask.shape[1])

    for i in range(mask.shape[0]):
        for j in range(i+1, mask.shape[0]):
            for t in range(1,mask.shape[1]):
                distance = torch.abs(comparison_vectors[t-1][i] - comparison_vectors[t-1][j]).to(device)
                pc = reg(distance)
                pcs[i][j][t] = pc
                pcs[j][i][t] = pc

    for sample_num in range(mask.shape[0]):
        for t in range(mask.shape[1]):
            if mask[sample_num, t] == 0:
                pc_s = pcs[sample_num, :, t]
                idx_sorted = torch.argsort(pc_s, descending=True)
                imp_val = 0
                n_t = 0
                for idx in idx_sorted[:2]:
                    imp_val += ground_truth[t][idx]
                    n_t += 1
                imp_val /= n_t
                ground_truth[t][sample_num] = imp_val

    return ground_truth
