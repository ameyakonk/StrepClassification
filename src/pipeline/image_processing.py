from src.common.lib import *
from src.common.dir import *
from src.common.var import *
from src.utils.utils import *
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')
class ImageProcessing:
    def __init__(self):
        pass  

    def white_balance(self, img):
        img = np.array(img).astype(float)

        ## average value of each channel
        avg_red = np.mean(img[:, :, 0])
        avg_green = np.mean(img[:, :, 1])
        avg_blue = np.mean(img[:, :, 2])
        
        # gray value calculation
        avg_gray = (avg_red + avg_green + avg_blue) / 3
        
        # Scaling each channel
        img[:, :, 0] *= (avg_gray / avg_red)
        img[:, :, 1] *= (avg_gray / avg_green)
        img[:, :, 2] *= (avg_gray / avg_blue)
        
        # Clip values to [0, 255]
        img = np.clip(img, 0, 255).astype(np.uint8)     #the factor can cause values to go beyond 255, hence bounding it.
        return Image.fromarray(img)
    
    def apply_clahe(self, img):
        img_np = np.array(img)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b))
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))

    def visualize_correction(self, image_path):
        original = Image.open(image_path).convert("RGB")
        
        # white balance function
        corrected = self.white_balance(original)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(original)
        axes[0].set_title("original")
        axes[0].axis('off')
        
        axes[1].imshow(corrected)
        axes[1].set_title("white balanced")
        axes[1].axis('off')
        
        plt.show()
    
    def get_dataset_stats(self, dataloader):
        pixel_sum = torch.zeros(3)
        pixel_squared_sum = torch.zeros(3)
        batches = 0
        
        for img, _, _ in dataloader:
            pixel_sum += torch.mean(img, dim=[0, 2, 3])              ## averages accross [batch, height, width]
            pixel_squared_sum += torch.mean(img**2, dim=[0, 2, 3])   ## squared averages accross [batch, height, width]
            batches += 1

        mean = pixel_sum/batches
        std = (pixel_squared_sum / batches - mean ** 2) ** 0.5        ## std = (E[X**2] - E[x]**2)**0.5
        logging.info(f"Mean: {mean}") 
        logging.info(f"Std: {std}")
        return mean, std



if __name__ == "__main__":

    obj = ImageProcessing()
    df = read_csv(UPDATED_CSV_FILE)
    # for idx in range(len(df)):
    #     img_vis = df.iloc[idx, 0]
    #     obj.visualize_correction(os.path.join(IMG_DIR, img_vis))
    #     break
