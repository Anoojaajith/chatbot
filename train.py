import json
from nltk_utils import tokenize,stem,bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from model import NeuralNet

if __name__ == '__main__':
    with open('intents.json','r') as f:
        intents = json.load(f)

all_words = []
tags = [] 
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w= tokenize(pattern)
        all_words.extend(w) 
        xy.append((w, tag))

ignore_words = ['?','!','.',',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

x_train =[]
y_train = []
for (pattern_sentence,tag) in xy:
    bag = bag_of_words(pattern_sentence,all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples


batch_size = 8
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
input_size = len(x_train[0])
print(f"Input size: {input_size}, Vocabulary size: {len(all_words)}")
print(f"Output size: {output_size}, Tags: {tags}")
num_epochs = 1000


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model = NeuralNet(input_size,hidden_size,output_size).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device, dtype=torch.long)

        #forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        #backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch +1)% 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss{loss.item():.4f}')

print(f'final loss, loss{loss.item():.4f}')
#save the data
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags


} 

file = "data.pth"
torch.save(data,file)

print(f'training complete. file saved to {file}')

