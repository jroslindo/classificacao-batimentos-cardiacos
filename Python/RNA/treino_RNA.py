from Lib import * 
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
### modelo da rede neural e parametros
model = ANN()
model.cuda()
model.train()


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.00055310) # 0.00015822  0.00055310
data = torch.load('data.pt')
target = torch.load('target.pt')
# print(data.shape)
# print(data[:-300].shape)
# print(data[-300:].shape)

entrada, entrada_validacao , respostas_vetor, respostas_vetor_validacao = train_test_split(data[:-300], target[:-300], test_size=0.1, random_state=10)
entrada.requires_grad_()

grafico_treino = []
grafico_validacao = []
grafico_x     = []

print("Começando o treino")
running_loss = 0.0

for epoch in range(70):  # loop over the dataset multiple times
    print("epoca " + str(epoch))
    anterior = model
    model.train()
    running_loss = 0.0
    for i in range(len(entrada)):
        entrada_aux = entrada[i].unsqueeze(0)

        optimizer.zero_grad()

        outputs = model(entrada_aux.unsqueeze(0))
        loss = criterion(outputs.unsqueeze(0), respostas_vetor[i].unsqueeze(0))
        # loss = criterion(outputs.unsqueeze(0), respostas_vetor[i].unsqueeze(0)) #respostas_vetor[i].unsqueeze(0)

        if i == 0:
            loss.backward(retain_graph=True)
        else:
            loss.backward(retain_graph=True)

        optimizer.step()
        running_loss += loss.item()

    erro_treino = (running_loss/len(entrada))*100
    grafico_treino.append(erro_treino)
    print(grafico_treino[-1])

    #VALIDANDOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
    running_loss = 0.0
    model.eval()
    certos = 0
    for i in range (len(entrada_validacao)):

        outputs = model(entrada_validacao[i].unsqueeze(0).unsqueeze(0))
        predicted = torch.max(outputs.unsqueeze(0), 1)


        # loss = criterion(outputs.unsqueeze(0), respostas_vetor_validacao[i].unsqueeze(0))
        # running_loss += loss.item()

        if predicted[1][0].item() == respostas_vetor_validacao[i].item():
            certos += 1

    erro_validacao = (1-certos/len(entrada_validacao))*100

    if epoch > 1 and grafico_validacao[-1] < erro_validacao and erro_treino < erro_validacao: #and (abs(grafico_validacao[-1] - erro_validacao) > 0.5) 
        grafico_validacao.append((1-certos/len(entrada_validacao))*100)
        grafico_x.append(epoch)
        print(grafico_validacao[-1])
        break

    grafico_validacao.append((1-certos/len(entrada_validacao))*100)
    grafico_x.append(epoch)
    print(grafico_validacao[-1])    
    print("---------------\n")


plt.plot(grafico_x, grafico_treino, grafico_x, grafico_validacao)
plt.show()

print('Finished Training:  ' + str(grafico_treino[-1]) + " --Salvando o modelo")

torch.save(anterior.state_dict(), "net.pth")




# exit()






entrada_teste = data[-300:]
resposta_teste = target[-300:]



print("validando: \n\n")

model.load_state_dict(torch.load("net.pth"))
model.eval()

desconhecido = 0
certos = 0
errados = 0
resposta = 0

arq = open("log.txt", "w")

for i in range (len(entrada_teste)):

    outputs = model(entrada_teste[i].unsqueeze(0).unsqueeze(0))
    predicted = torch.max(outputs.unsqueeze(0), 1)
    arq.write(str(outputs) + "\n")
    # print (torch.max(outputs.unsqueeze(0), 1))
    arq.write("previsto: " + str(predicted[1][0].item()) + "\n")
    arq.write("resposta: " + str(resposta_teste[i].item())+ "\n")
    # input("enter")

    if predicted[1][0].item() == resposta_teste[i].item():
        arq.write("certo"+ "\n")
        certos += 1
    else:        
        arq.write("errado"+ "\n")
        errados += 1
    
    arq.write("--------------\n\n")





print("total: " + str(len(entrada_teste))+ "\n")
print("Certos: " + str(certos) + " Porcentagem: " + str(certos/len(entrada_teste))+ "\n")
print("Errados: " + str(errados) + " Porcentagem: " + str(errados/len(entrada_teste))+ "\n")
print("Desconhecido: " + str(desconhecido) + " Porcentagem: " + str(desconhecido/len(entrada_teste))+ "\n")