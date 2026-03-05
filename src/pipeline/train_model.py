from src.common.lib import *
from src.common.dir import *
from src.common.var import *
from src.common.model import ClassificationModel
from src.utils.utils import *
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

STAGE_NAME = "Train Model"

class ModelTraining:
    def __init__(self, train_loader, val_loader):
        self.classifier = ClassificationModel()
        self.strep_model = self.classifier.model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_auroc = float(0)
        self.train_loss = float(0)
        self.train_acc = float(0)
        self.model_name = MODEL_NAME
        self.auroc_metric = BinaryAUROC().to(DEVICE)

    def model_train(self):
        create_directory(MODEL_DIR)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.strep_model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)
        epochs = EPOCHS

        for epoch in range(epochs):
            self.strep_model.train()
            
            train_loss = 0.0
            preds_train = 0.0
            correct_train = 0.0
            total_train = 0.0

            for images, features, labels in self.train_loader:
                images, features, labels = images.to(DEVICE), features.to(DEVICE), labels.to(DEVICE).float()

                optimizer.zero_grad()
                outputs = self.strep_model(images, features.float()).view(-1) ## [1, batch_size] -> [batch_size]
                loss = criterion(outputs, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.strep_model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()

                preds_train = (outputs > 0).float()
                correct_train += (preds_train == labels).sum().item()
                total_train += labels.size(0)

            train_accuracy = 100 * correct_train / total_train

            self.auroc_metric.reset()
            with torch.no_grad():
                val_loss = 0.0
                correct = 0.0
                total = 0.0
                self.strep_model.eval()
                for images, features, labels in self.val_loader:
                    images, features, labels = images.to(DEVICE), features.to(DEVICE).float(), labels.to(DEVICE).float()
                    
                    outputs = self.strep_model(images, features).view(-1)
                    val_loss += criterion(outputs, labels).item()

                    probs = torch.sigmoid(outputs)
                    self.auroc_metric.update(probs, labels.int())
                    
                    # probabilities to binary prediction
                    preds = (outputs > 0).float()
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            avg_train_loss = train_loss / len(self.train_loader)
            avg_val_loss = val_loss / len(self.val_loader)
            accuracy = 100 * correct / total

            epoch_auroc = self.auroc_metric.compute()

            if epoch_auroc > self.best_auroc:
                self.best_auroc = epoch_auroc
                self.train_acc = train_accuracy
                self.train_loss = avg_train_loss
                self.classifier.save_model(self.strep_model.state_dict(), self.model_name)
                logging.info(f"Model saved with ROC-AUC: {epoch_auroc:.4f}")
            
            print()
            logging.info(f"Epoch [{epoch+1}/{epochs}] "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Train Acc: {train_accuracy:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} | "
                    f"Val Acc: {accuracy:.2f}%  |"
                    f"Val ROC-AUC: {epoch_auroc:.4f}")
            print()
        
        print()
        logging.info(f"Train Loss: {self.train_loss:.4f} | "
                    f"Train Acc: {self.train_acc:.4f} | "
            f"Best ROC-AUC: {self.best_auroc:.4f}")
        print()