import os
import nltk
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import models
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import sys

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

#nltk.download("punkt")
#nltk.download('punkt_tab')

# Dataset class
class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, captions, vocab, transform=None):
        self.image_dir = image_dir
        self.captions = captions
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions.iloc[idx]["caption"]
        image_name = self.captions.iloc[idx]["image"]
        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        caption_tokens = self.vocab.tokenize(caption)
        caption_indices = [self.vocab.word2idx[token] for token in caption_tokens]
        return image, torch.tensor(caption_indices)


# Vocabulary class
class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}

    def build_vocab(self, sentences):
        freq = {}
        for sentence in sentences:
            if not isinstance(sentence, str):
                continue  # Skip non-string captions
            for word in nltk.word_tokenize(sentence.lower()):
                freq[word] = freq.get(word, 0) + 1

        for word, count in freq.items():
            if count >= self.freq_threshold:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word


    def tokenize(self, sentence):
        tokens = nltk.word_tokenize(sentence.lower())
        return ["<SOS>"] + [token if token in self.word2idx else "<UNK>" for token in tokens] + ["<EOS>"]

# Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images):
        features = self.features(images)
        features = self.pool(features)
        features = features.view(features.size(0), -1)
        return features

# Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, vocab_size, encoder_dim, num_heads, num_layers, dropout):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, embed_size))
        self.feature_projection = nn.Linear(encoder_dim, embed_size)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=2048, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, features, captions):
        features = self.feature_projection(features)
        embeddings = self.embedding(captions) + self.positional_encoding[:, :captions.size(1), :].to(captions.device)
        features = features.unsqueeze(1).repeat(1, captions.size(1), 1).permute(1, 0, 2)
        embeddings = embeddings.permute(1, 0, 2)
        decoder_output = self.transformer_decoder(embeddings, features)
        predictions = self.fc(decoder_output)
        return predictions.permute(1, 0, 2)

# Evaluation with BLEU
def evaluate_bleu(encoder, decoder, dataloader, vocab):
    encoder.eval()
    decoder.eval()
    bleu_scores = []

    with torch.no_grad():
        for images, captions in tqdm(dataloader, desc="Evaluating BLEU"):
            images = torch.stack(images).to(device)
            features = encoder(images)

            for idx, caption in enumerate(captions):
                generated_caption = []
                feature = features[idx].unsqueeze(0)
                decoder_input = torch.tensor([vocab.word2idx["<SOS>"]]).unsqueeze(0).to(device)

                for _ in range(20):
                    predictions = decoder(feature, decoder_input)
                    predicted_idx = predictions[:, -1, :].argmax(dim=1)
                    generated_caption.append(predicted_idx.item())
                    if predicted_idx.item() == vocab.word2idx["<EOS>"]:
                        break
                    decoder_input = torch.cat([decoder_input, predicted_idx.unsqueeze(0)], dim=1)

                generated_caption = [vocab.idx2word[idx] for idx in generated_caption if idx not in {vocab.word2idx["<PAD>"], vocab.word2idx["<SOS>"], vocab.word2idx["<EOS>"]}]
                reference_caption = [vocab.idx2word[idx] for idx in caption.tolist() if idx not in {vocab.word2idx["<PAD>"], vocab.word2idx["<SOS>"], vocab.word2idx["<EOS>"]}]
                bleu = sentence_bleu([reference_caption], generated_caption, smoothing_function=SmoothingFunction().method1)
                bleu_scores.append(bleu)

    return sum(bleu_scores) / len(bleu_scores)

def collate_fn(batch):
    images, captions = zip(*batch)
    return list(images), list(captions)


def main():
    # Paths
    image_dir = "./FlickerData/Images"
    caption_file = "./FlickerData/captions.txt"

    # Vocabulary and Dataset
    vocab = Vocabulary(freq_threshold=5)

    # Reload the captions file
    captions = pd.read_csv(caption_file, sep="\t", names=["image_caption"])

    # Split the combined column into 'image' and 'caption'
    captions_split = captions["image_caption"].str.split(",", n=1, expand=True)
    captions["image"] = captions_split[0].str.strip()  # Extract and clean image names
    captions["caption"] = captions_split[1].str.strip()  # Extract and clean captions

    # Drop the original combined column
    captions = captions.drop(columns=["image_caption"])

    # Remove rows with missing or non-existent image files
    captions["image_path"] = captions["image"].apply(lambda x: os.path.join(image_dir, x))
    captions = captions[captions["image_path"].apply(os.path.exists)]

    # Drop the now unnecessary 'image_path' column
    captions = captions.drop(columns=["image_path"])


    # Verify the first few rows
    print(captions.head())


    vocab.build_vocab(captions["caption"].tolist())

    train_captions, test_captions = train_test_split(captions, test_size=0.1, random_state=42)
    train_captions, val_captions = train_test_split(train_captions, test_size=0.1, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = Flickr8kDataset(image_dir=image_dir, captions=train_captions, vocab=vocab, transform=transform)
    val_dataset = Flickr8kDataset(image_dir=image_dir, captions=val_captions, vocab=vocab, transform=transform)
    test_dataset = Flickr8kDataset(image_dir=image_dir, captions=test_captions, vocab=vocab, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=8)

    # Model
    encoder = Encoder().to(device)
    decoder = TransformerDecoder(embed_size=256, vocab_size=len(vocab.word2idx), encoder_dim=2048, num_heads=8, num_layers=6, dropout=0.1).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<PAD>"])
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)

    # Training
    num_epochs = 3
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        train_loss = 0

        for images, captions in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
            images = torch.stack(images).to(device)
            captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=vocab.word2idx["<PAD>"]).to(device)

            optimizer.zero_grad()
            features = encoder(images)
            predictions = decoder(features, captions[:, :-1])
            loss = criterion(predictions.reshape(-1, predictions.shape[2]), captions[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader)}")

    # Evaluate
    test_bleu = evaluate_bleu(encoder, decoder, test_loader, vocab)
    print(f"BLEU Score on Test Set: {test_bleu:.4f}")

if __name__ == "__main__":
    main()
