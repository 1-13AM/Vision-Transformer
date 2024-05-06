from PIL import Image
import torchvision.transforms as transforms
def transform(IMAGE_HEIGHT, IMAGE_WIDTH):
    def any_to_rgb(img):
        return img.convert('RGB')

    transformation = transforms.Compose([
                     transforms.Lambda(any_to_rgb),
                     transforms.ToTensor(),
                     transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225]),
                        ])
    
    return transformation
