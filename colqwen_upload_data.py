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

# Initialize the model and processor
model_name = "impactframes/colqwen2-v0.1"
model = ColQwen2.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"}
)
processor = ColQwen2Processor.from_pretrained(model_name)
model = model.eval()

# Sample PDFs
sample_pdfs = [
    {
        "title": "ConocoPhillips Sustainability Highlights - Nature (24-0976)",
        "url": "https://static.conocophillips.com/files/resources/24-0976-sustainability-highlights_nature.pdf",
    },
    {
        "title": "ConocoPhillips Managing Climate Related Risks",
        "url": "https://static.conocophillips.com/files/resources/conocophillips-2023-managing-climate-related-risks.pdf",
    },
    {
        "title": "ConocoPhillips 2023 Sustainability Report",
        "url": "https://static.conocophillips.com/files/resources/conocophillips-2023-sustainability-report.pdf",
    },
    {
        "title": "WaterPipes",
        "url": "https://www.arct.cam.ac.uk/sites/www.arct.cam.ac.uk/files/p_33campbell.pdf",
    },
    {
        "title": "Water Falls",
        "url": "https://abbeyroadprimary.co.uk/wp-content/uploads/2020/07/Geography-Rivers-session-2-Waterfalls-Input.pdf",
    },
]

# Helper function to resize images
def resize_image(image, max_height=800):
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
        batch_size=1,
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
        base_64_image = get_base64_image(resize_image(image, 640))
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
        vespa_feed
