from Lib import * 

model = ANN()
# print(model)
# exit()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

X_train, y_train = load_mfcc_GPU()


epochs = 100
loss_arr = []
for i in range(epochs):
   y_hat = model.forward(X_train)
   loss = criterion(y_hat, y_train)
   loss_arr.append(loss)
 
   if i % 10 == 0:
       print(f'Epoch: {i} Loss: {loss}')
 
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()

print("terminou de treinar")