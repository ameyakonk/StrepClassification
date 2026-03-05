from src.common.lib import *
from src.common.dir import *
from src.common.var import *
from src.common.model import ClassificationModel
from src.utils.utils import *
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

STAGE_NAME = "Evaluate Model"

class ModelEvaluation:
    def __init__(self,  test_loader):
        self.classifier = ClassificationModel()
        self.strep_model = self.classifier.model
        self.test_loader = test_loader
        self.auroc_metric = BinaryAUROC().to(DEVICE)
    
    def model_evaluate(self):
        criterion = nn.BCEWithLogitsLoss()
        self.classifier.load_model(MODEL_NAME)
        self.auroc_metric.reset()
        with torch.no_grad(): 
            test_loss = 0.0
            correct = 0.0
            total = 0.0
            self.strep_model.eval()
            for images, features, labels in self.test_loader:
                images, features, labels = images.to(DEVICE), features.to(DEVICE).float(), labels.to(DEVICE).float()
                
                outputs = self.strep_model(images, features).view(-1)
                test_loss += criterion(outputs, labels).item()
                
                # Convert probabilities to binary predictions (0 or 1)
                preds = (outputs > 0).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                probs = torch.sigmoid(outputs)
                self.auroc_metric.update(probs, labels.int())

            # Print progress
            avg_test_loss = test_loss / len(self.test_loader)
            accuracy = 100 * correct / total
            epoch_auroc = self.auroc_metric.compute()
            
            print()
            logging.info( f"Test Loss: {avg_test_loss:.4f} | "
                    f"Test Acc: {accuracy:.2f}% |"
                    f"Val ROC-AUC: {epoch_auroc:.4f}")
            print()
