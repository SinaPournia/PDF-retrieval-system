import torch
import requests
import numpy as np
from pdf2image import convert_from_path
from pypdf import PdfReader
import base64
from torch.utils.data import DataLoader
from tqdm import tqdm
from io import BytesIO
from colpali_engine.models import ColQwen2, ColQwen2Processor
from vespa.application import Vespa
from vespa.io import VespaResponse
import asyncio
import json
from config import Settings  
settings = Settings()
# Initialize the model and processor using settings
model_name = settings.model_name  # Get model name from settings
# Automatically set device_map to "cuda" if GPU is available, otherwise "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# Load the model with the appropriate device and dtype
model = ColQwen2.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device  # Automatically select GPU if available
)
processor = ColQwen2Processor.from_pretrained(model_name)
model = model.eval()

def load_pdfs_from_json(json_file_path):
    with open(json_file_path, 'r') as f:
        return json.load(f)

# Path to the JSON file containing PDF details
json_file_path = 'pdfs.json'
sample_pdfs = load_pdfs_from_json(json_file_path)

# Helper function to resize images using settings
def resize_image(image, max_height=settings.image_resize):  # Use image resize from settings
    width, height = image.size
    if height > max_height:
        ratio = max_height / height
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        return image.resize((new_width, new_height))
    return image

# Download PDF
def download_pdf(url):
    response = requests.get(url)
    if response.status_code == 200:
        return BytesIO(response.content)
    else:
        raise Exception(f"Failed to download PDF: Status code {response.status_code}")

# Convert PDF to images and extract text
def get_pdf_images(pdf_url):
    pdf_file = download_pdf(pdf_url)
    temp_file = "temp.pdf"
    with open(temp_file, "wb") as f:
        f.write(pdf_file.read())
    reader = PdfReader(temp_file)
    page_texts = []
    for page_number in range(len(reader.pages)):
        page = reader.pages[page_number]
        text = page.extract_text()
        page_texts.append(text)
    images = convert_from_path(temp_file)
    assert len(images) == len(page_texts)
    return (images, page_texts)

# Process each PDF and extract images and texts
for pdf in sample_pdfs:
    page_images, page_texts = get_pdf_images(pdf['url'])
    pdf['images'] = page_images
    pdf['texts'] = page_texts

# Create embeddings for images in each PDF
for pdf in sample_pdfs:
    page_embeddings = []
    dataloader = DataLoader(
        pdf['images'],
        batch_size=settings.batch_size,  
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x),
    )
    print(f"Created DataLoader for {len(pdf['images'])} images.")

    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
            page_embeddings.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
            print(f"Current page embeddings: {len(page_embeddings)} pages embedded.")
    
    pdf['embeddings'] = page_embeddings

# Convert image to base64
def get_base64_image(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return str(base64.b64encode(buffered.getvalue()), "utf-8")

# Prepare data for Vespa
vespa_feed = []
for pdf in sample_pdfs:
    url = pdf['url']
    title = pdf['title']
    for page_number, (page_text, embedding, image) in enumerate(zip(pdf['texts'], pdf['embeddings'], pdf['images'])):
        base_64_image = get_base64_image(resize_image(image, settings.image_resize))  # Use dynamic image resize
        embedding_dict = dict()
        for idx, patch_embedding in enumerate(embedding):
            binary_vector = np.packbits(np.where(patch_embedding > 0, 1, 0)).astype(np.int8).tobytes().hex()
            embedding_dict[idx] = binary_vector      
        page = {
            "id": hash(url + str(page_number)),
            "url": url,
            "title": title,
            "page_number": page_number,
            "image": base_64_image,
            "text": page_text,
            "embedding": embedding_dict
        }
        vespa_feed.append(page)


async def feed_vespa_pages(appLocal, vespa_feed):
    async with appLocal.asyncio(connections=1, total_timeout=180) as session:
        for page in tqdm(vespa_feed):
            response: VespaResponse = await session.feed_data_point(
                data_id=page['id'], fields=page, schema=settings.vespa_app_name
            )
            if not response.is_successful():
                print(response.json())

async def main():
    app = Vespa(url=settings.vespa_url) 
    await feed_vespa_pages(app, vespa_feed)

if __name__ == "__main__":
    asyncio.run(main())