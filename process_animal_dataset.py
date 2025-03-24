import os
import numpy as np
import librosa
import torch
import clip
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from PIL import Image
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm

# Configure data paths
DATA_ROOT = "datasets/newdata"
PIC_DIR = os.path.join(DATA_ROOT, "picture")
SOUND_DIR = os.path.join(DATA_ROOT, "sound")
EMBEDDING_DIR = os.path.join(DATA_ROOT, "embeddings")
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)
wav_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)

# Predefined 10 animal categories
ANIMALS = ["bird", "cat", "chicken", "cow", "dog", "frog", "lion", "monkey", "pig", "sheep"]

def load_and_sort_files(directory, extension):
    """Load and sort files by animal category + index to ensure data alignment"""
    files = sorted([f for f in os.listdir(directory) if f.endswith(extension)])
    return files

def extract_image_features():
    """Extract CLIP features for all images, ensuring per-animal storage"""
    all_embeddings = []
    animal_embeddings = {animal: [] for animal in ANIMALS}

    filenames = load_and_sort_files(PIC_DIR, ".jpg")

    for fname in tqdm(filenames, desc="Extracting Image Features"):
        img_path = os.path.join(PIC_DIR, fname)
        animal = fname.split("_")[0]  # Extract animal name from filename

        if animal not in ANIMALS:
            print(f"Skipping unrecognized file: {fname}")
            continue

        image = clip_preprocess(Image.open(img_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            img_features = clip_model.encode_image(image).cpu().numpy().flatten()
        
        all_embeddings.append(img_features)
        animal_embeddings[animal].append(img_features)

    # Save per-animal embeddings
    for animal, embeddings in animal_embeddings.items():
        np.save(os.path.join(EMBEDDING_DIR, f"{animal}_img.npy"), np.array(embeddings))
    
    np.save(os.path.join(EMBEDDING_DIR, "pic_embeddings.npy"), np.array(all_embeddings))
    print("Image embeddings saved!")

def extract_sound_features():
    """Extract Wav2Vec2 features for all audio files, ensuring per-animal storage"""
    all_embeddings = []
    animal_embeddings = {animal: [] for animal in ANIMALS}

    filenames = load_and_sort_files(SOUND_DIR, ".wav")

    for fname in tqdm(filenames, desc="Extracting Sound Features"):
        sound_path = os.path.join(SOUND_DIR, fname)
        animal = fname.split("_")[0]  # Extract animal name from filename

        if animal not in ANIMALS:
            print(f"Skipping unrecognized file: {fname}")
            continue

        wav, sr = librosa.load(sound_path, sr=16000)
        input_values = wav_processor(wav, return_tensors="pt", sampling_rate=sr).input_values.to(device)

        with torch.no_grad():
            sound_features = wav_model(input_values).last_hidden_state.mean(dim=1).cpu().numpy().flatten()

        all_embeddings.append(sound_features)
        animal_embeddings[animal].append(sound_features)

    # Save per-animal embeddings
    for animal, embeddings in animal_embeddings.items():
        np.save(os.path.join(EMBEDDING_DIR, f"{animal}_sound.npy"), np.array(embeddings))

    np.save(os.path.join(EMBEDDING_DIR, "sound_embeddings.npy"), np.array(all_embeddings))
    print("Sound embeddings saved!")

def generate_labels():
    """Generate label vectors to ensure data alignment"""
    labels = []
    for animal in ANIMALS:
        labels.extend([animal] * 10)  # 10 samples per animal
    
    np.save(os.path.join(EMBEDDING_DIR, "labels.npy"), np.array(labels))
    print("Labels saved!")

def visualize_tsne(embeddings, labels, title):
    """Reduce dimensionality using t-SNE and visualize clusters"""
    print(f"[DEBUG] Embeddings shape: {embeddings.shape}")
    print(f"[DEBUG] Labels shape before filtering: {len(labels)}")

    # Ensure labels match embedding shape
    if embeddings.shape[0] != len(labels):
        labels = labels[: embeddings.shape[0]]
        print(f"[DEBUG] Labels shape after correction: {len(labels)}")

    tsne = TSNE(n_components=2, perplexity=10, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    
    unique_labels = np.unique(labels)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_idx[label] for label in labels])

    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels[: tsne_results.shape[0]], palette="tab10", alpha=0.7)
    plt.title(title)
    plt.legend(title="Animals", loc="best")
    plt.show()

# Run data preprocessing
if __name__ == "__main__":
    extract_image_features()
    extract_sound_features()
    generate_labels()

    print("\nFeature extraction complete! Data saved in:", EMBEDDING_DIR)

    # Load embeddings and labels
    pic_embeddings = np.load(os.path.join(EMBEDDING_DIR, "pic_embeddings.npy"))
    sound_embeddings = np.load(os.path.join(EMBEDDING_DIR, "sound_embeddings.npy"))
    labels = np.load(os.path.join(EMBEDDING_DIR, "labels.npy"))

    # Check dimensions
    print(f"Picture embeddings: {pic_embeddings.shape}")
    print(f"Sound embeddings: {sound_embeddings.shape}")
    print(f"Labels: {labels.shape}")

    # Visualize pre-GCCA embeddings
    visualize_tsne(pic_embeddings, labels, "t-SNE Visualization of Image Embeddings")
    visualize_tsne(sound_embeddings, labels, "t-SNE Visualization of Sound Embeddings")

    print("\nPre-GCCA visualization complete!")
