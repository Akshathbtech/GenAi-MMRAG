import os
from pathlib import Path
import requests
import torch
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct, ScoredPoint
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize Qdrant client
client = QdrantClient(url=QDRANT_URL, timeout=150)

# Define paths
data_path = Path("data_wiki")
image_path = Path("data_wiki/images")
user_data_path = Path("user_data")
user_image_path = user_data_path / "images"
data_path.mkdir(exist_ok=True, parents=True)
image_path.mkdir(exist_ok=True, parents=True)
user_data_path.mkdir(exist_ok=True, parents=True)
user_image_path.mkdir(exist_ok=True, parents=True)

# Create Qdrant collections if they don't exist
if not client.collection_exists("text_collection_0"):
    client.create_collection("text_collection_0", vectors_config=VectorParams(size=512, distance=Distance.COSINE))
if not client.collection_exists("image_collection_0"):
    client.create_collection("image_collection_0", vectors_config=VectorParams(size=512, distance=Distance.COSINE))

# Function to add user-uploaded text to Qdrant
def add_user_text_to_qdrant(user_text):
    inputs = processor(text=user_text, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
    text_embedding = model.get_text_features(**inputs).detach().cpu().numpy().flatten().tolist()
    user_point = PointStruct(
        id=str(uuid.uuid4()),
        vector=text_embedding,
        payload={"type": "text", "origin": "user", "content": user_text[:100] + "..."}
    )
    client.upsert(collection_name="text_collection_0", points=[user_point])
    st.write("User text added to the database successfully!")

# Function to add user-uploaded image to Qdrant
def add_user_image_to_qdrant(image):
    if image.mode == "RGBA":
        image = image.convert("RGB")
    
    image_id = str(uuid.uuid4())
    image_file = user_image_path / f"user_image_{image_id}.jpg"
    image.save(image_file)
    
    inputs = processor(images=image, return_tensors="pt").to(device)
    image_embedding = model.get_image_features(**inputs).detach().cpu().numpy().flatten().tolist()
    image_point = PointStruct(
        id=image_id,
        vector=image_embedding,
        payload={"type": "image", "origin": "user", "filename": str(image_file)}
    )
    client.upsert(collection_name="image_collection_0", points=[image_point])
    st.write("User image added to the database successfully!")

# Function to retrieve similar texts
def retrieve_similar_texts(query_text, top_k=3):
    query_inputs = processor(text=query_text, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
    query_embedding = model.get_text_features(**query_inputs).detach().cpu().numpy().flatten().tolist()
    text_results = client.search(collection_name="text_collection_0", query_vector=query_embedding, limit=top_k)
    
    results_with_content = []
    for result in text_results:
        if isinstance(result, ScoredPoint) and result.payload.get("type") == "text":
            title = result.payload.get("title", "User Text")
            content = result.payload.get("content", "Content not found.")
            results_with_content.append({
                "title": title,
                "score": result.score,
                "content": content
            })
    
    return results_with_content

# Function to retrieve similar images
def retrieve_similar_images(query_text, top_k=5):
    query_inputs = processor(text=query_text, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
    query_embedding = model.get_text_features(**query_inputs).detach().cpu().numpy().flatten().tolist()
    image_results = client.search(collection_name="image_collection_0", query_vector=query_embedding, limit=top_k)
    
    retrieved_image_paths = []
    for result in image_results:
        if isinstance(result, ScoredPoint) and result.payload.get("type") == "image":
            image_path = Path(result.payload.get("filename", ""))
            if image_path.exists():
                retrieved_image_paths.append(image_path)
    
    return retrieved_image_paths


# Streamlit UI setup
st.title("Multi-Modal Retrieval System")
st.write("Retrieve related texts and images from Wikipedia articles or add your own content.")

# Text input for user to add custom content
user_input = st.text_area("Enter your own article or text to add:", "")
if st.button("Add My Text"):
    if user_input.strip():
        add_user_text_to_qdrant(user_input)
    else:
        st.write("Please enter some text.")

# Image input for user to add a custom image
uploaded_image = st.file_uploader("Upload an image:", type=["jpg", "png"])
if st.button("Add My Image") and uploaded_image is not None:
    image = Image.open(uploaded_image)
    add_user_image_to_qdrant(image)

# Query existing or added content
query = st.text_input("Search articles and images:", "What is the significance of the Kesavananda Bharati case?")
if st.button("Retrieve Results"):
    text_results = retrieve_similar_texts(query)
    image_paths = retrieve_similar_images(query)

    # Display text results
    st.write("### Text Retrieval Results:")
    if text_results:
        for result in text_results:
            st.write(f"**Title:** {result['title']}, **Score:** {result['score']}")
            st.write(f"**Content:** {result['content']}")
            st.write("---")

    # Display image results
    st.write("### Image Retrieval Results:")
    if image_paths:
        for path in image_paths:
            st.image(str(path), caption="Retrieved Image")
    else:
        st.write("No images to display.")
