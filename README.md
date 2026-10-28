# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:Janani saraswathi S

### Register Number:212225230110
```
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("C:\\Users\\acer\\Downloads\\Exp-1.csv")
df

x = df[["Input"]].values
y = df[["Output"]].values

x_test, x_train, y_test, y_train = train_test_split(x,y,test_size=0.4,random_state=42)

scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()

x_train = scaler1.fit_transform(x_train)
x_test = scaler2.fit_transform(x_test)


x_train1 = torch.FloatTensor(x_train)
x_test1 = torch.FloatTensor(x_test)
y_train1 = torch.FloatTensor(y_train)
y_test1 = torch.FloatTensor(y_test)

class neuralnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1,20),
            nn.ReLU(),
            nn.Linear(20,10),
            nn.ReLU(),
            nn.Linear(10,1)
        )
    def forward(self,x):
        return self.network(x)


model = neuralnet()
criterion =nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

epochs = 1000
loss_history = []

for i in range(epochs):
    optimizer.zero_grad()
    pred = model(x_train1)
    loss = criterion(pred,y_train1)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if i%100==0:
        print(f"Epoch{i}:",loss.item())


with torch.no_grad():
    pred=model(x_test1)
    loss=criterion(pred,y_test1)
    print(loss.item())

model.eval()

input = torch.tensor(scaler1.transform([[22]]), dtype=torch.float32)

with torch.no_grad():
    pred = model(input)

print(pred.item())

import matplotlib.pyplot as plt

plt.plot(range(1, epochs + 1), loss_history)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Epochs")
plt.grid(True)
plt.show()

**
### Dataset Information
<img width="570" height="591" alt="image" src="https://github.com/user-attachments/assets/6eac646e-3cf4-49ac-a440-c9e16e0f8de0" />


### OUTPUT

<img width="482" height="497" alt="image" src="https://github.com/user-attachments/assets/ef5ee7b8-2ca8-45ff-aa34-00785cc0e9d3" />


### Training Loss Vs Iteration Plot

<img width="555" height="411" alt="image" src="https://github.com/user-attachments/assets/793133e5-8bdb-4571-ab70-2016b7dd45e1" />

### New Sample Data Prediction

<img width="558" height="80" alt="image" src="https://github.com/user-attachments/assets/1a31da10-4420-4f91-848e-272742135acc" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
