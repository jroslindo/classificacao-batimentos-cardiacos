from Lib import * 

### modelo da rede neural e parametros
model = ANN()
model.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)




# scaler = GradScaler()

entrada, respostas_vetor = load_mfcc_GPU()

for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(entrada.unsqueeze(0))
    print(outputs)
    
    loss = criterion(outputs, respostas_vetor)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    
    # print(loss)
    running_loss = 0.0

print('Finished Training')