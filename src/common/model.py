from src.common.lib import *
from src.common.dir import *
from src.common.var import *
from src.utils.utils import *
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

class StrepClassifier(nn.Module):
    def __init__(self, num_features, num_classes=1):
        super(StrepClassifier, self).__init__()
        
        # Pre trained ResNet18
        resnet = models.resnet18(weights='DEFAULT')

        # freeze till the second last layer
        for param in resnet.parameters():
            param.requires_grad = False
        
        # unfreeze the last layer
        for param in resnet.layer4.parameters():
            param.requires_grad = True
        
        self.image_branch = nn.Sequential(*list(resnet.children())[:-1])

        self.img_projection = nn.Sequential(
            nn.Linear(512, 32),
            nn.BatchNorm1d(32), 
            nn.ReLU()
        )

        # MLP for numerical data
        # self.feature_branch = nn.Sequential(
        #     nn.Linear(num_features, 16),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(),
        # )
        
        # # Combine image embeddings (32) + feature embeddings (16)
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, num_classes), 
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, images, features):
        # Process image
        img_out = self.image_branch(images)
        img_out = torch.flatten(img_out, 1) # [batch, 512]
        img_out = self.img_projection(img_out)# [batch, 32]
       
        # # Process features
        # feat_out = self.feature_branch(features) # [batch, 16]
        
        # # Concatenate branches
        # combined = torch.cat((img_out, feat_out), dim=1) # [batch, 48]

        return self.classifier(img_out) # [batch, 1]

class ClassificationModel:
    def __init__(self):    
        self.model = StrepClassifier(num_features = 7, num_classes = 1).to(DEVICE)

    def save_model(self, state_dict, model_name):
        torch.save(state_dict, os.path.join(MODEL_DIR, model_name))

    def load_model(self, model_name):
        # return self.model.load_state_dict(torch.load(os.path.join(MODEL_DIR, model_name)))
        weights = torch.load(os.path.join(MODEL_DIR, model_name), weights_only=True)
        return self.model.load_state_dict(weights)
        