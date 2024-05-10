import torch.nn
import matplotlib.pyplot as plt
import seaborn as sns

vit_full_losses = torch.load('models/vit_full/model-losses.pth')

sns.set(style='darkgrid')

plt.figure(figsize=(10, 6))

plt.plot(range(vit_full_losses['num_epochs']), vit_full_losses['validation_losses'], label='Validation loss', color='red')
plt.plot(range(vit_full_losses['num_epochs']), vit_full_losses['train_losses'], label='Train loss', color='blue')
plt.title('Vision Transformers Full Model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('ViT-Full-loss_curves.png')