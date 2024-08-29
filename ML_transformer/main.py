
from torch.utils.data import random_split, DataLoader
from GF_dataset import *
from ViT_Classifier import *
from training import start_training

if __name__ == "__main__":
    device = "cpu"

    cell_radius = 100  # in meters
    N = 100
    K = 8  # N total users, K active users
    P = 1
    freq = 2  # in GHz
    SNR = 1000  # in dB
    Lp = 12  # Pilot sequence length L << N -> 12
    M = 8
    root = os.path.abspath("..")
    path = os.path.join(root, "pilots", "ICBP_" + str(Lp) + "_100.mat")
    A, _ = get_ICBP_pilots(path, N, K)

    dataset = GF_dataset(A, K, M, P, SNR, cell_radius, freq, 1000)

    train_size = int(0.95 * len(dataset))
    test_size = len(dataset) - train_size

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=True)
    num_classes = dataset.getNumClasses()
    dim = (N,M)

    ViT_Classifier = VisionTransformer(
        img_size=dim, patch_size=(5,4), in_chans=3,embed_dim=256, depth=8, num_heads=8, num_classes=num_classes,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))

    lr = 0.01
    epochs = 10
    pos_weight = 12
    reg_weight = 0.1
    base_loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight)) #This takes care of the sigmoid already and is numerically more stable according to copilot
    loss_function = CustomLoss(base_loss_function, K, reg_weight=reg_weight)
    optimizer = torch.optim.Adam(ViT_Classifier.parameters(), lr=lr, weight_decay=0.01)

    start_training(epochs, train_loader, test_loader, ViT_Classifier, loss_function, optimizer, device)

