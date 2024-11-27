import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
from collections import Counter
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as T

import pandas as pd

from PIL import Image
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.models as models
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as T
from tqdm import tqdm
from torch.utils.data import random_split
from torchtext.data.metrics import bleu_score

from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
spacy_eng = spacy.load("en_core_web_sm")



class EarlyStopping:
    def __init__(self, patience=8, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = np.inf
        self.couter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    


class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)


    def forward(self, images):
        features = self.resnet(images)                                    #(batch_size,2048,7,7)
        features = features.permute(0, 2, 3, 1)                           #(batch_size,7,7,2048)
        features = features.view(features.size(0), -1, features.size(-1)) #(batch_size,49,2048)
        return features

class Attention(nn.Module):
    def __init__(self, encoder_dim,decoder_dim,attention_dim):
        super(Attention, self).__init__()

        self.attention_dim = attention_dim

        self.W = nn.Linear(decoder_dim,attention_dim)
        self.U = nn.Linear(encoder_dim,attention_dim)

        self.A = nn.Linear(attention_dim,1)




    def forward(self, features, hidden_state):
        u_hs = self.U(features)     #(batch_size,num_layers,attention_dim)
        w_ah = self.W(hidden_state) #(batch_size,attention_dim)

        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1)) #(batch_size,num_layers,attemtion_dim)

        attention_scores = self.A(combined_states)         #(batch_size,num_layers,1)
        attention_scores = attention_scores.squeeze(2)     #(batch_size,num_layers)


        alpha = F.softmax(attention_scores,dim=1)          #(batch_size,num_layers)

        attention_weights = features * alpha.unsqueeze(2)  #(batch_size,num_layers,features_dim)
        attention_weights = attention_weights.sum(dim=1)   #(batch_size,num_layers)

        return alpha,attention_weights


class DecoderRNN(nn.Module):
    def __init__(self,embed_size, vocab_size, attention_dim,encoder_dim,decoder_dim,drop_prob=0.3):
        super().__init__()

        #save the model param
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim

        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.attention = Attention(encoder_dim,decoder_dim,attention_dim)


        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.lstm_cell = nn.LSTMCell(embed_size+encoder_dim,decoder_dim,bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)


        self.fcn = nn.Linear(decoder_dim,vocab_size)
        self.drop = nn.Dropout(drop_prob)



    def forward(self, features, captions):

        #vectorize the caption
        embeds = self.embedding(captions)

        # Initialize LSTM state
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)

        #get the seq length to iterate
        seq_length = len(captions[0])-1 #Exclude the last one
        batch_size = captions.size(0)
        num_features = features.size(1)

        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, seq_length,num_features).to(device)

        for s in range(seq_length):
            alpha,context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))

            output = self.fcn(self.drop(h))

            preds[:,s] = output
            alphas[:,s] = alpha


        return preds, alphas

    def generate_caption(self,features,max_len=20,vocab=None):
        # Inference part
        # Given the image features generate the captions

        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)

        alphas = []

        #starting input
        word = torch.tensor(vocab.stoi['<SOS>']).view(1,-1).to(device)
        embeds = self.embedding(word)


        captions = []

        for i in range(max_len):
            alpha,context = self.attention(features, h)


            #store the apla score
            alphas.append(alpha.cpu().detach().numpy())

            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.fcn(self.drop(h))
            output = output.view(batch_size,-1)


            #select the word with most val
            predicted_word_idx = output.argmax(dim=1)

            #save the generated word
            captions.append(predicted_word_idx.item())

            #end if <EOS detected>
            if vocab.itos[predicted_word_idx.item()] == "<EOS>":
                break

            #send generated word as the next caption
            embeds = self.embedding(predicted_word_idx.unsqueeze(0))

        #covert the vocab idx to words and return sentence
        return [vocab.itos[idx] for idx in captions],alphas


    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c


class EncoderDecoder(nn.Module):
    def __init__(self,embed_size, vocab_size, attention_dim,encoder_dim,decoder_dim,drop_prob=0.3):
        super().__init__()
        self.encoder = EncoderCNN()
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            vocab_size = vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim
        )

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs


class CapsCollate:
    """
    Collate to apply the padding to the captions with dataloader
    """
    def __init__(self,pad_idx,batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first

    def __call__(self,batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs,dim=0)

        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        return imgs,targets

def save_model(model,num_epochs, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, trainloss, valloss):
    model_state = {
        'num_epochs':num_epochs,
        'embed_size':embed_size,
        'vocab_size':vocab_size,
        'attention_dim':attention_dim,
        'encoder_dim':encoder_dim,
        'decoder_dim':decoder_dim,
        'train_loss': trainloss,
        'val_loss': valloss,
        'state_dict':model.state_dict()
    }

    torch.save(model_state,f'attention_model_state_{num_epochs}.pth')



class Vocabulary:
    def __init__(self,freq_threshold):
        #setting the pre-reserved tokens int to string tokens
        self.itos = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}

        #string to int tokens
        #its reverse dict self.itos
        self.stoi = {v:k for k,v in self.itos.items()}

        self.freq_threshold = freq_threshold

    def __len__(self): return len(self.itos)

    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

                #add the word to the vocab if it reaches minum frequecy threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self,text):
        """ For each word in the text corresponding index token for that word form the vocab built as list """
        tokenized_text = self.tokenize(text)
        return [ self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text ]


class FlickrDataset(Dataset):
    """
    FlickrDataset
    """
    def __init__(self,root_dir,captions_file,transform=None,freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        #Get image and caption colum from the dataframe
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        #Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())


    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_location = os.path.join(self.root_dir,img_name)
        img = Image.open(img_location).convert("RGB")

        #apply the transfromation to the image
        if self.transform is not None:
            img = self.transform(img)

        #numericalize the caption text
        caption_vec = []
        caption_vec += [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]

        return img, torch.tensor(caption_vec)
    



def main():
    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor()
    ])

    early_stopping = EarlyStopping()


    dataset = FlickrDataset("FlickerData/Images/", "FlickerData/captions.txt", transform=transform)

    def show_image(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    img, caps = dataset[10]
    #show_image(img,"Image")
    print("Token:",caps)
    print("Sentence:")
    print([dataset.vocab.itos[token] for token in caps.tolist()])

    total_size = len(dataset)

    # Define split proportions
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    # Calculate split sizes
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size  # Ensure total matches


    # Split the dataset
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])


    BATCH_SIZE = 32
    NUM_WORKER = 4

    pad_idx = dataset.vocab.stoi["<PAD>"]



    train_loader = DataLoader(
            dataset=train_set,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKER,
            shuffle=True,
            collate_fn=CapsCollate(pad_idx=pad_idx,batch_first=True)
        )

    val_loader = DataLoader(
            dataset=val_set,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKER,
            shuffle=True,
            collate_fn=CapsCollate(pad_idx=pad_idx,batch_first=True)
        )

    test_loader = DataLoader(
            dataset=test_set,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKER,
            shuffle=True,
            collate_fn=CapsCollate(pad_idx=pad_idx,batch_first=True)
        )

    embed_size=300
    vocab_size = len(dataset.vocab)
    attention_dim=256
    encoder_dim=2048
    decoder_dim=512
    learning_rate = 3e-4

    print("Hyperparameters done")

    model = EncoderDecoder(
            embed_size=300,
            vocab_size = len(dataset.vocab),
            attention_dim=256,
            encoder_dim=2048,
            decoder_dim=512
        ).to(device)

    print("Model done")

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Optimizers Done")

    num_epochs = 25
    print_every = 100

    train_loss = []
    val_loss_list = []


    best_loss = np.inf
    patience_counter = 0

    print("train parameters done")

    # for epoch in range(1,num_epochs+1):
    #         train_loss_all = 0
    #         val_loss_all = 0
    #         counter = 0
    #         for image, captions in tqdm(train_loader):
    #         #for idx, (image, captions) in enumerate(iter(train_loader)):
    #             image,captions = image.to(device),captions.to(device)

    #             # Zero the gradients.
    #             optimizer.zero_grad()

    #             # Feed forward
    #             outputs,attentions = model(image, captions)

    #             # Calculate the batch loss.
    #             targets = captions[:,1:]
    #             loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
    #             train_loss_all += loss.item()
    #             # Backward pass.
    #             loss.backward()

    #             # Update the parameters in the optimizer.
    #             optimizer.step()
    #             counter += 1

    #             if (counter+1)%print_every == 0:


    #                 #generate the caption
    #                 model.eval()
    #                 with torch.no_grad():
    #                     for val_image, val_captions in val_loader:
    #                         val_image,val_captions = val_image.to(device),val_captions.to(device)
    #                         # Feed forward
    #                         val_outputs,val_attentions = model(val_image, val_captions)
    #                         # Calculate the batch loss.
    #                         val_targets = val_captions[:,1:]
    #                         val_loss = criterion(val_outputs.view(-1, vocab_size), val_targets.reshape(-1))
    #                         val_loss_all += val_loss.item()
    #                     dataiter = iter(val_loader)
    #                     img,_ = next(dataiter)
    #                     features = model.encoder(img[0:1].to(device))
    #                     caps,alphas = model.decoder.generate_caption(features,vocab=dataset.vocab)
    #                     caption = ' '.join(caps)
    #                     #show_image(img[0],title=caption)

    #                 model.train()
    #         train_loss.append(train_loss_all)
    #         val_loss_list.append(val_loss_all)

    #         #Early stopping
    #         early_stopping(val_loss)

    #         if early_stopping.early_stop:
    #             print("Early stopping triggered")
    #             break



    #         #save the latest model
    #         print("Train: Epoch: {} loss: {:.5f}".format(epoch,loss.item()))
    #         print("Val: Epoch: {} loss: {:.5f}".format(epoch,loss.item()))
    #         save_model(model,epoch, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, train_loss, val_loss_list)


    checkpoint = torch.load('attention_model_state_13.pth')
    model = EncoderDecoder(embed_size=checkpoint['embed_size'], vocab_size=checkpoint['vocab_size'], 
                attention_dim=checkpoint['attention_dim'], encoder_dim=checkpoint['encoder_dim'], 
                decoder_dim=checkpoint['decoder_dim']).to(device)
    
    model.load_state_dict(checkpoint['state_dict'])
    bleu1, bleu2, bleu3 = evaluate_model_bleu(model, test_loader, dataset)
    
    print("-----------------------------")
    print("Blue scores of the own:")
    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-2: {bleu2:.4f}")
    print(f"BLEU-3: {bleu3:.4f}")

    # train_loss = checkpoint['train_loss']
    # val_loss = checkpoint['val_loss']

    # plt.figure(figsize=(10, 6))
    # plt.plot(train_loss, label='Training Loss')
    # plt.plot(val_loss, label='Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss Over Epochs')
    # plt.legend()
    # plt.grid()
    # plt.show()

    # dataiter = iter(test_loader)
    # images,_ = next(dataiter)

    # img = images[0].detach().clone()
    # img1 = images[0].detach().clone()
    # caps,alphas = get_caps_from(img.unsqueeze(0), model, dataset)

    # plot_attention(img1, caps, alphas)

    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

    # ADD YOUR CODE HERE
    model.eval()
    hypotheses = []
    references = []

    with torch.no_grad():  # No gradient calculation needed for evaluation
        #for batch in tqdm(test_loader):
        for images, target_captions in tqdm(test_loader):
            # Assuming `image` and `captions` keys in the batch dictionary
            #images, target_captions = batch['images'], batch['captions']
            images, target_captions = images.to(device), target_captions.to(device)
            # Preprocess images
            pixel_values = feature_extractor(images=list(images), return_tensors="pt", do_rescale=False).pixel_values.to(device)

            # Generate captions
            output_ids = model.generate(pixel_values, max_length=16, num_beams=4)

            # Decode generated captions
            predicted_captions = [
                tokenizer.decode(output_id, skip_special_tokens=True) for output_id in output_ids
            ]

            # Process reference captions for BLEU or other evaluation
            for i, caption in enumerate(target_captions):
                reference_tokens = [
                    dataset.vocab.itos[idx.item()]
                    for idx in caption
                    if idx.item() not in {dataset.vocab.stoi['<PAD>'], dataset.vocab.stoi['<EOS>']}
                ]
                references.append([reference_tokens])
                hypotheses.append(predicted_captions[i].split())  # Tokenize predicted captions

    
    bleu1 = bleu_score(hypotheses, references, max_n=1, weights=[1.0])
    bleu2 = bleu_score(hypotheses, references, max_n=2, weights=[0.5, 0.5])
    bleu3 = bleu_score(hypotheses, references, max_n=3, weights=[1/3, 1/3, 1/3])

    print("-----------------------------")
    print("Blue scores of the state-of-art:")
    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-2: {bleu2:.4f}")
    print(f"BLEU-3: {bleu3:.4f}")



def preprocess_images(images, feature_extractor):
    # Transform images to the required format
    pixel_values = feature_extractor(images=list(images), return_tensors="pt").pixel_values
    return pixel_values.to(device)
import sys
#generate caption
def get_caps_from(features_tensors, model, dataset):
    #generate the caption
    model.eval()
    with torch.no_grad():
        features = model.encoder(features_tensors.to(device))
        caps,alphas = model.decoder.generate_caption(features,vocab=dataset.vocab)
        caption = ' '.join(caps)
        show_image(features_tensors[0],title=caption)

    return caps,alphas
def show_image(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


#Show attention
def plot_attention(img, result, attention_plot):
    #untransform
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224
    img[2] = img[2] * 0.225
    img[0] += 0.485
    img[1] += 0.456
    img[2] += 0.406

    img = img.numpy().transpose((1, 2, 0))
    temp_image = img

    fig = plt.figure(figsize=(15, 15))

    len_result = len(result)
    for l in range(len_result):
        temp_att = attention_plot[l].reshape(7,7)

        ax = fig.add_subplot(len_result//2,len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.7, extent=img.get_extent())


    plt.tight_layout()
    plt.show()




def evaluate_model_bleu(model, test_loader, dataset):
    model.eval()  # Ensure the model is in evaluation mode
    hypotheses = []
    references = []
    
    with torch.no_grad():  # No gradient calculation needed for evaluation
        for image, captions in tqdm(test_loader):
            # Move data to the appropriate device
            image, captions = image.to(device), captions.to(device)

            # Get model outputs
            outputs, attentions = model(image, captions)
            
            # Targets: Remove <start> token
            targets = captions[:, 1:]

            # Get predictions by taking the argmax over the vocabulary dimension
            predicted_indices = torch.argmax(outputs, dim=2)  # Shape: (batch_size, seq_len)

            for i in range(targets.size(0)):  # Process each item in the batch
                # Convert predicted indices to words
                predicted_tokens = [
                    dataset.vocab.itos[idx.item()] 
                    for idx in predicted_indices[i]
                    if idx.item() not in {dataset.vocab.stoi['<PAD>'], dataset.vocab.stoi['<EOS>']}
                ]

                # Convert target indices to words
                reference_tokens = [
                    [
                        dataset.vocab.itos[idx.item()] 
                        for idx in targets[i]
                        if idx.item() not in {dataset.vocab.stoi['<PAD>'], dataset.vocab.stoi['<EOS>']}
                    ]
                ]  # Wrap in another list to support multiple references

                # Append to BLEU lists
                hypotheses.append(predicted_tokens)
                references.append(reference_tokens)

    # Calculate BLEU scores
    bleu1 = bleu_score(hypotheses, references, max_n=1, weights=[1.0])
    bleu2 = bleu_score(hypotheses, references, max_n=2, weights=[0.5, 0.5])
    bleu3 = bleu_score(hypotheses, references, max_n=3, weights=[1/3, 1/3, 1/3])

    return bleu1, bleu2, bleu3

if __name__ == "__main__":
    main()