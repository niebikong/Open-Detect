import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import net


class OpenDetectNet(nn.Module):
    def __init__(self, arch='resnet18', channel=3, latent_dim=128, n_classes=10, temp_inter=0.1, temp_intra=1, init=True):
        super(OpenDetectNet, self).__init__()
        self.arch = arch
        self.channel = channel
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.temp_inter = temp_inter
        self.temp_intra = temp_intra
        self.encoder, self.decoder = net(self.arch, self.channel, self.latent_dim)
        self.prototypes = nn.Parameter(torch.randn(self.n_classes, self.latent_dim).cuda(), requires_grad=True)
        if init:
            nn.init.kaiming_normal_(self.prototypes)

    def sampler(self, mu, logvar):
        # Reparameterization trick for sampling latent variable z
        std = torch.exp(0.5 * logvar)
        if self.training:
            z = mu + std * torch.randn_like(std)
        else:
            z = mu
        return z

    def distance(self, latent_z, prototypes):
        # Compute squared Euclidean distance between latent_z and prototypes
        matrixA_square = torch.sum(latent_z ** 2, 1, keepdim=True)
        matrixB_square = torch.sum(prototypes ** 2, 1).unsqueeze(0)
        product_A_B = torch.matmul(latent_z, prototypes.t())
        return matrixA_square + matrixB_square - 2 * product_A_B

    def kl_div_to_prototypes(self, mean, logvar, prototypes):
        # KL divergence between N(mu, sigma) and all prototype Gaussians N(mu_w, I)
        kl_div = self.distance(mean, prototypes) + torch.sum(logvar.exp() - logvar - 1, dim=1, keepdim=True)
        return 0.5 * kl_div

    def forward(self, x):
        mu, logvar, lateral_z = self.encoder(x)
        latent_z = self.sampler(mu, logvar)
        dist = self.distance(latent_z, self.prototypes)
        kl_div = self.kl_div_to_prototypes(mu, logvar, self.prototypes)
        recon_x = self.decoder(latent_z, lateral_z)
        return latent_z, dist, kl_div, recon_x

    def loss(self, x, y):
        latent_z, dist, kl_div, x_recon = self.forward(x)
        # Predict class by nearest prototype
        dist_reshape = dist.view(len(x), self.n_classes, 1)
        dist_class_min, _ = torch.min(dist_reshape, dim=2)
        _, preds = torch.min(dist_class_min, dim=1)
        # Distance and KL to ground-truth class prototypes
        y_one_hot = F.one_hot(y, num_classes=self.n_classes).bool()
        dist_y = dist[y_one_hot].view(len(dist), 1)
        kl_div_y = kl_div[y_one_hot].view(len(kl_div), 1)
        q_w_z_y = F.softmax(-dist_y / self.temp_intra, dim=1)
        # Reconstruction loss (MSE)
        rec_loss = F.mse_loss(x_recon, x)
        # Conditional prior KL loss
        q_w_z_y = torch.clamp(q_w_z_y, min=1e-7)
        kld_loss = torch.mean(torch.sum(q_w_z_y * kl_div_y, dim=1))
        # Entropy loss
        ent_loss = torch.mean(torch.sum(q_w_z_y * torch.log(q_w_z_y * self.n_classes), dim=1))
        # Discriminative loss (logsumexp)
        LSE_all_dist = torch.logsumexp(-dist / self.temp_inter, 1)
        LSE_target_dist = torch.logsumexp(-dist_y / self.temp_inter, 1)
        dis_loss = torch.mean(LSE_all_dist - LSE_target_dist)
        loss = {'dis': dis_loss, 'rec': rec_loss, 'kld': kld_loss, 'ent': ent_loss}
        return latent_z, x_recon, preds, loss


def train_model(model, args, train_loader, epoch, optimizer):
    # Training loop for one epoch
    num_epochs = args.epoch
    model.train()
    print('Current learning rate is {}'.format(optimizer.param_groups[0]['lr']))
    print('Epoch {}/{}'.format(epoch + 1, num_epochs))
    print('*' * 70)
    train_corrects = 0
    running_loss = {}
    for i, (image, label) in enumerate(train_loader):
        image, label = image.cuda(), label.cuda()
        optimizer.zero_grad()
        _, _, preds, loss = model.loss(image, label)
        total_loss = args.lamda * (loss['rec'] + loss['kld'] + loss['ent']) + (1 - args.lamda) * loss['dis']
        loss['total'] = total_loss
        total_loss.backward()
        optimizer.step()
        for k in loss.keys():
            running_loss[k] = loss.get(k, 0).item() + running_loss.get(k, 0)
        train_corrects += torch.sum(preds == label.data)
    train_acc = train_corrects.item() / len(train_loader.dataset)
    train_loss = {k: running_loss.get(k, 0) / len(train_loader) for k in running_loss.keys()}
    print('Train corrects: {} Train samples: {} Train accuracy: {}'.format(
        train_corrects, len(train_loader.dataset), train_acc))
    print('Train loss: {:.3f}= {}*[rec({:.3f}) + kld({:.3f}) + ent({:.3f})] + (1-{})*dis({:.3f})'.format(
        train_loss['total'], args.lamda, train_loss['rec'], train_loss['kld'],
        train_loss['ent'], args.lamda, train_loss['dis']))


def validate_model(model, args, val_loader, epoch):
    # Validation loop for one epoch
    model.eval()
    val_corrects = 0.0
    val_running_loss = {'total': 0.0, 'rec': 0.0, 'kld': 0.0, 'ent': 0.0, 'dis': 0.0}
    for image, label in val_loader:
        with torch.no_grad():
            image, label = image.cuda(), label.cuda()
            latent_z, x_recon, preds, loss = model.loss(image, label)
            total_loss = args.lamda * (loss['rec'] + loss['kld'] + loss['ent']) + (1 - args.lamda) * loss['dis']
            loss['total'] = total_loss
            for k in loss.keys():
                val_running_loss[k] = loss.get(k, 0).item() + val_running_loss.get(k, 0)
            val_corrects += torch.sum(preds == label.data)
    val_acc = val_corrects / len(val_loader.dataset)
    val_loss = {k: val_running_loss.get(k, 0) / len(val_loader) for k in val_running_loss.keys()}
    print('Val corrects: {} Val samples: {} Val accuracy: {}'.format(
        val_corrects, len(val_loader.dataset), val_acc))
    print('Val loss: {:.3f}= {}*[rec({:.3f}) + kld({:.3f}) + ent({:.3f})] + (1-{})*dis({:.3f})'.format(
        val_loss['total'], args.lamda, val_loss['rec'], val_loss['kld'],
        val_loss['ent'], args.lamda, val_loss['dis']))
    print('*' * 70)
    return val_acc


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = OpenDetectNet('resnet18', 1, 128, 24, 0.1, 1)
    net.to(device)
    input = torch.randn(1, 1, 40, 40).to(device)
    print(net(input))
    
    # print(net.parameters)