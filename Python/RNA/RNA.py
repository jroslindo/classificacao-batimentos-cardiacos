from Lib import * 

### modelo da rede neural e parametros
model = ANN()
model.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.00015822, momentum=0.9)
entrada, respostas_vetor = load_mfcc_GPU()




# entrada = entrada[0].unsqueeze(0)


# running_loss = 0.0

# optimizer.zero_grad()

# outputs = model(entrada.unsqueeze(0))

# loss = criterion(outputs, respostas_vetor[0].unsqueeze(0))

# loss.backward()

# optimizer.step()

# running_loss += loss.item()

# print(running_loss)

running_loss = 0.0
for epoch in range(50):  # loop over the dataset multiple times
    
    for i in range(len(entrada)):
        entrada_aux = entrada[i].unsqueeze(0)    

        optimizer.zero_grad()

        outputs = model(entrada_aux.unsqueeze(0))

        loss = criterion(outputs, respostas_vetor[i].unsqueeze(0))

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    if (epoch%10==9):
        print(str(running_loss/(409*epoch)) + "\n---------------------\n")

    # input()
    




print('Finished Training:  ' + str(running_loss/(409*epoch)))