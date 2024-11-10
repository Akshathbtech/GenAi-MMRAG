import os
from pathlib import Path
import requests
import wikipedia
import urllib.request
import torch
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct, ScoredPoint
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
import shutil

load_dotenv()
# Retrieve Qdrant URL from environment variables
QDRANT_URL = os.getenv("QDRANT_URL")

# Set up device and load models
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Paths and Settings
data_path = Path("data_wiki")
image_path = Path("data_wiki/images")
MAX_IMAGES_PER_WIKI = 15
data_path.mkdir(exist_ok=True)
image_path.mkdir(exist_ok=True)

# Wikipedia Titles to Retrieve
wiki_titles = ["Nosferatu", "Chola Dynasty", "Dhananjaya Y. Chandrachud", "Kesavananda Bharati v. State of Kerala"]

# Download Wikipedia texts and images
def download_wikipedia_data():
    image_metadata_dict = {}
    image_uuid = 0
    
    for title in wiki_titles:
        # Download text
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "explaintext": True,
            },
        ).json()
        page = next(iter(response["query"]["pages"].values()))
        wiki_text = page["extract"]
        
        # Save text to local storage
        with open(data_path / f"{title}.txt", "w", encoding="utf-8") as fp:
            fp.write(wiki_text)
        
        # Download images
        images_per_wiki = 0
        try:
            page_py = wikipedia.page(title)
            list_img_urls = page_py.images
            for url in list_img_urls:
                if url.endswith(".jpg") or url.endswith(".png"):
                    image_uuid += 1
                    image_file_name = f"{title}_{url.split('/')[-1]}"
                    image_metadata_dict[image_uuid] = {
                        "filename": image_file_name,
                        "img_path": image_path / f"{image_uuid}.jpg"
                    }
                    urllib.request.urlretrieve(url, image_path / f"{image_uuid}.jpg")
                    images_per_wiki += 1
                    if images_per_wiki >= MAX_IMAGES_PER_WIKI:
                        break
        except Exception:
            continue

    return image_metadata_dict

# Initialize Qdrant Client to connect to the server
client = QdrantClient(url=QDRANT_URL, timeout=150)
client.recreate_collection("text_collection_0", vectors_config=VectorParams(size=512, distance=Distance.COSINE))
client.recreate_collection("image_collection_0", vectors_config=VectorParams(size=512, distance=Distance.COSINE))

# Load text data from files
def load_text_data(directory):
    documents = []
    for filepath in Path(directory).glob("*.txt"):
        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()
            documents.append({"text": text, "filename": filepath.name})
    return documents

# Function to generate embeddings and store in Qdrant
def generate_embeddings(image_metadata_dict):
    documents = load_text_data(data_path)
    
    # Generate text embeddings
    for i, doc in enumerate(documents):
        if doc["text"]:
            inputs = processor(text=doc["text"], return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
            text_embedding = model.get_text_features(**inputs).detach().cpu().numpy().flatten()
            point = PointStruct(id=i, vector=text_embedding.tolist(), payload={"type": "text", "title": doc["filename"]})
            client.upsert(collection_name="text_collection_0", points=[point])

    # Generate image embeddings
    for uuid, metadata in image_metadata_dict.items():
        image = Image.open(metadata["img_path"])
        inputs = processor(images=image, return_tensors="pt").to(device)
        image_embedding = model.get_image_features(**inputs).detach().cpu().numpy().flatten()
        point = PointStruct(id=uuid, vector=image_embedding.tolist(), payload={"type": "image", "filename": metadata["filename"]})
        client.upsert(collection_name="image_collection_0", points=[point])

# Download data and generate embeddings
image_metadata_dict = download_wikipedia_data()
generate_embeddings(image_metadata_dict)

# Updated retrieval function to fetch relevant information
def retrieve_similar_texts(query_text, top_k=3):
    query_inputs = processor(text=query_text, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
    query_embedding = model.get_text_features(**query_inputs).detach().cpu().numpy().flatten().tolist()
    text_results = client.search(collection_name="text_collection_0", query_vector=query_embedding, limit=top_k)
    
    # Extract text content for each result
    results_with_content = []
    for result in text_results:
        if isinstance(result, ScoredPoint) and result.payload.get("type") == "text":
            title = result.payload.get("title", "No Title")
            file_path = data_path / title
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                truncated_content = content[:500] + "..." if len(content) > 500 else content
            results_with_content.append({
                "title": title,
                "score": result.score,
                "content": truncated_content
            })
    
    return results_with_content

# Retrieval function for images
def retrieve_similar_images(query_text, top_k=5):
    query_inputs = processor(text=query_text, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
    query_embedding = model.get_text_features(**query_inputs).detach().cpu().numpy().flatten().tolist()
    image_results = client.search(collection_name="image_collection_0", query_vector=query_embedding, limit=top_k)
    
    # Retrieve image paths from local storage
    retrieved_image_paths = []
    for result in image_results:
        if isinstance(result, ScoredPoint) and result.payload.get("type") == "image":
            image_path = image_metadata_dict[result.id]["img_path"]
            retrieved_image_paths.append(image_path)
    
    return retrieved_image_paths

# Streamlit UI setup
st.title("Multi-Modal Retrieval System")
st.write("Retrieve related texts and images from Wikipedia articles.")

# Text Input for Query
query = st.text_input("Enter your query:", "What is the significance of the Kesavananda Bharati case?")
if st.button("Retrieve Results"):
    text_results = retrieve_similar_texts(query)
    image_paths = retrieve_similar_images(query)

    # Display text results with content
    st.write("### Text Retrieval Results:")
    if text_results:
        for result in text_results:
            st.write(f"**Title:** {result['title']}, **Score:** {result['score']}")
            st.write(f"**Content:** {result['content']}")
            st.write("---")  # Separator between results

    # Display image results
    st.write("### Image Retrieval Results:")
    if image_paths:
        for path in image_paths:
            st.image(str(path), caption="Retrieved Image")
    else:
        st.write("No images to display.")
