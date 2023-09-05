import torch
from torchvision import transforms
from PIL import Image
from model import DeblurCNN  # deblurring model class

# Load the pre-trained model
model = DeblurCNN()
model.load_state_dict(torch.load('deblur_model.pth'))  # Load the trained model weights
model.eval()  # Set the model to evaluation mode

# Load and preprocess the input image
input_image_path = 'input_image.jpg'  # Replace with the path to your input image
input_image = Image.open(input_image_path)
input_tensor = transforms.ToTensor()(input_image).unsqueeze(0)

# Perform inference
with torch.no_grad():
    output_tensor = model(input_tensor)

# Convert the output tensor to a PIL image
output_image = transforms.ToPILImage()(output_tensor.squeeze(0).cpu())

# Save or display the deblurred image
output_image.save('output_image.jpg')  # Save the deblurred image
output_image.show()  # Display the deblurred image