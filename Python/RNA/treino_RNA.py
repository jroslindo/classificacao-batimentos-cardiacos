from Lib import * 

### modelo da rede neural e parametros
model = ANN()
model.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0015822, betas=[0.000076253698849,0.000076253698849], weight_decay=0.85565561 )
entrada, respostas_vetor = load_mfcc_GPU()


print("Começando o treino")
running_loss = 0.0

for epoch in range(50):  # loop over the dataset multiple times
    print(epoch)
    for i in range(len(entrada)):
        entrada_aux = entrada[i].unsqueeze(0)

        optimizer.zero_grad()

        outputs = model(entrada_aux.unsqueeze(0))

        loss = criterion(outputs, respostas_vetor[i].unsqueeze(0))

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    if (epoch % 10 == 9):
        print(str(running_loss/(len(entrada)*epoch)) + "\n---------------------\n")



print('Finished Training:  ' + str(running_loss/(len(entrada)*epoch)) + " --Salvando o modelo")

torch.save(model.state_dict(), "net.pth")