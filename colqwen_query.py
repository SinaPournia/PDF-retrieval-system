import asyncio
from torch.utils.data import DataLoader
import torch
from vespa.io import VespaQueryResponse
from colpali_engine.models import ColQwen2, ColQwen2Processor
from vespa.application import Vespa
import webbrowser
import os
from config import settings  # Import settings
import json

with open("queries.json", "r") as f:
    queries = json.load(f)["queries"]

for query in queries:
    print(query)


# Initialize the model and processor using settings
model_name = settings.colpali_model_name  # Get model name from settings
print(torch.cuda.is_available())  # Check if CUDA is available
model = ColQwen2.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map=settings.device_map  # Use settings for torch dtype and device
)
processor = ColQwen2Processor.from_pretrained(model_name)
model = model.eval()  # Set model to evaluation mode


# Create a DataLoader to process the queries
dataloader = DataLoader(
    queries,
    batch_size=settings.batch_size,  # Use batch size from settings
    shuffle=False,
    collate_fn=lambda x: processor.process_queries(x),
)

# Generate embeddings for each query using the model
qs = []
for batch_query in dataloader:
    with torch.no_grad():  # Disable gradient calculation
        batch_query = {k: v.to(model.device) for k, v in batch_query.items()}  # Move data to model's device
        embeddings_query = model(**batch_query)  # Get query embeddings
        qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))  # Store embeddings on the CPU

# Function to save query results as an HTML file and display it
def save_query_results_as_html(query, response, hits=5, file_name="results.html"):
    # Extract search time and result count from the response
    query_time = response.json.get('timing', {}).get('searchtime', -1)
    query_time = round(query_time, 2)
    count = response.json.get('root', {}).get('fields', {}).get('totalCount', 0)
    
    # Start building the HTML content
    html_content = f'<h3>Query text: \'{query}\', query time {query_time}s, count={count}, top results:</h3>'
    
    # Loop through top results and add them to the HTML content
    for i, hit in enumerate(response.hits[:hits]):  
        title = hit['fields']['title']
        url = hit['fields']['url']
        page = hit['fields']['page_number']
        image = hit['fields']['image']
        score = hit['relevance']
        
        # Add information about each result
        html_content += f'<h4>PDF Result {i + 1}</h4>'
        html_content += f'<p><strong>Title:</strong> <a href="{url}">{title}</a>, page {page+1} with score {score:.2f}</p>'
        html_content += f'<img src="data:image/png;base64,{image}" style="max-width:100%;">'
    
    # Get the absolute path for the file
    abs_file_path = os.path.abspath(file_name)
    
    # Save the HTML content to the file
    with open(abs_file_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    # Open the HTML file in the default web browser
    webbrowser.open(f"file://{abs_file_path}")
    
    print(f"Results saved to: {abs_file_path}")

# Initialize Vespa application with local instance
app = Vespa(url=settings.vespa_url)  # Use dynamic URL and port from settings

# Define an asynchronous function to execute queries
async def main():
    # Open a session with Vespa using asyncio
    async with app.asyncio(connections=1, total_timeout=120) as session:
        for idx, query in enumerate(queries):
            # Prepare query embedding from the model output
            query_embedding = {k: v.tolist() for k, v in enumerate(qs[idx])}
            
            # Execute the Vespa query with embeddings and additional parameters
            response: VespaQueryResponse = await session.query(
                yql="select title,url,image,page_number from pdf_page where userInput(@userQuery)",  # YQL query
                ranking=settings.ranking_profile_name,  # Use ranking profile from settings
                userQuery=query,
                timeout=120,  # Set a timeout
                hits=3,  # Limit results to 3 hits
                body={
                    "input.query(qt)": query_embedding,  # Embed query in the request body
                    "presentation.timing": True  # Request timing information
                },
            )
            
            # Ensure the response was successful
            assert response.is_successful(), f"Query failed for: {query}"
            
            # Save and display the query results in HTML format
            save_query_results_as_html(query, response, file_name=f"results_{idx}.html")

# Entry point for the script
if __name__ == "__main__":
    asyncio.run(main())  # Run the main asynchronous function
