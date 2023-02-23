import torch
import timm

# Define model (which extends the NN module)
class custom(torch.nn.Module):

    # Initialize
    def __init__(self):
        super(custom, self).__init__()

        # Load backbone
        backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0, global_pool = '')
        #o = backbone(torch.randn(2, 3, 224, 224))
        #print(f'Unpooled shape: {o.shape}')

        # Freeze the backbone weights
        for param in backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze last blocks
        for b in range(7,12):
            for param in backbone.blocks[b].parameters():
                param.requires_grad = True

        # Remove classifier (i.e. extract feature detection layers)
        self.features = backbone

        # Add simple prediction head
        self.head = torch.nn.Conv2d(in_channels=768, out_channels=1, kernel_size=1, stride=1, padding=0)

    # Forward
    def forward(self, x):
        b, c, h, w = x.shape
        x = self.features(x)
        x = x[:,1:,:] # Remove CLS token!!!!
        x = x.reshape(b, -1, 14, 14).contiguous()
        x = self.head(x)
        return x

#FIN