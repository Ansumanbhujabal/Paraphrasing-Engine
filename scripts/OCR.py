from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import os

# Path to your PDF file
pdf_path = "/teamspace/studios/this_studio/Paraphrasing-Engine/pdf/Devops_AWS_Certificate.pdf"
# Path where images will be temporarily saved
output_image_folder = 'pdf_images/'

# Ensure the folder for images exists
os.makedirs(output_image_folder, exist_ok=True)

# Convert PDF to a list of images (one image per page)
images = convert_from_path(pdf_path)

# Perform OCR on each image
extracted_text = ""
for i, image in enumerate(images):
    # Save image temporarily
    image_path = f'{output_image_folder}page_{i + 1}.png'
    image.save(image_path, 'PNG')

    # Extract text from the image using pytesseract
    text = pytesseract.image_to_string(Image.open(image_path))

    # Append the extracted text to the result
    extracted_text += text + "\n"

# Clean up temporary images if needed
for image_path in os.listdir(output_image_folder):
    os.remove(os.path.join(output_image_folder, image_path))

# Print or process the extracted text
print(extracted_text)

# Optionally save the text to a file
with open('extracted_text.txt', 'w') as text_file:
    text_file.write(extracted_text)
