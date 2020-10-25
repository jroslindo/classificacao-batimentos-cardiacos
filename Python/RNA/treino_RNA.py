from Lib import * 
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



### modelo da rede neural e parametros
model = ANN()
model.cuda()
model.train()
model.dropout = 0.85565561

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.00015822, momentum=0.8) # 0.00015822  0.00055310
data = torch.load('data.pt')
target = torch.load('target.pt')


entrada, entrada_validacao , respostas_vetor, respostas_vetor_validacao = train_test_split(data, target, test_size=0.2, random_state=20)
entrada.requires_grad_()

grafico_treino = []
grafico_validacao = []
grafico_x     = []

entrada_teste = entrada_validacao[324:]
resposta_teste = respostas_vetor_validacao[324:]

entrada_validacao = entrada_validacao[:324]
respostas_vetor_validacao = respostas_vetor_validacao[:324]
# print("Começando o treino")


# historico_rede = []

# for epoch in range(100):  # loop over the dataset multiple times
#     print("epoca " + str(epoch))
#     # anterior = model
    
#     aux_treino = treino()
#     aux_treino.epoca = epoch
    
#     # TREINANDOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
#     model.train()
#     model.dropout = 0.85565561
#     running_loss = 0.0
#     for i in range(len(entrada)):
#         entrada_aux = entrada[i].unsqueeze(0)

#         optimizer.zero_grad()

#         outputs = model(entrada_aux.unsqueeze(0))
#         loss = criterion(outputs.unsqueeze(0), respostas_vetor[i].unsqueeze(0))
#         # loss = criterion(outputs.unsqueeze(0), respostas_vetor[i].unsqueeze(0)) #respostas_vetor[i].unsqueeze(0)

#         if i == 0:
#             loss.backward(retain_graph=True)
#         else:
#             loss.backward(retain_graph=True)

#         optimizer.step()
#         running_loss += loss.item()

#     erro_treino = (running_loss/len(entrada))*100
#     grafico_treino.append(erro_treino)    
#     aux_treino.erro_treino = erro_treino
#     print(grafico_treino[-1])

#     #VALIDANDOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
#     running_loss = 0.0
#     model.eval()
#     certos = 0
#     for i in range (len(entrada_validacao)):

#         outputs = model(entrada_validacao[i].unsqueeze(0).unsqueeze(0))
#         predicted = torch.max(outputs.unsqueeze(0), 1)

#         if predicted[1][0].item() == respostas_vetor_validacao[i].item():
#             certos += 1

#     erro_validacao = (1-certos/len(entrada_validacao))*100
#     aux_treino.erro_validacao = erro_validacao
#     aux_treino.rede = model

#     grafico_validacao.append(erro_validacao)
#     grafico_x.append(epoch)
#     print(grafico_validacao[-1])
#     historico_rede.append(aux_treino)
#     print("---------------\n")


# plt.plot(grafico_x, grafico_treino, grafico_x, grafico_validacao)
# plt.show()

# print('Finished Training:  ' + str(grafico_treino[-1]) + " --Salvando o modelo")



# aux_menor = treino()
# aux_menor.erro_validacao = 100
# for erro in historico_rede:
#     if erro.erro_validacao < aux_menor.erro_validacao:
#         aux_menor = erro

# print("menor erro treino: ")
# print(aux_menor.erro_treino)
# print("menor erro validacao: ")
# print(aux_menor.erro_validacao)
# print("melhor epoca: ")
# print(aux_menor.epoca)

# torch.save(aux_menor.rede.state_dict(), "net.pth")



print("validando: \n\n")

model.load_state_dict(torch.load("net-088-2.pth"))
model.eval()

desconhecido = 0
certos = 0
errados = 0
resposta = 0
falso_positivo = 0
falso_negativo = 0


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

        if predicted[1][0].item() == 0 and resposta_teste[i].item() == 1:
            falso_negativo += 1
        elif predicted[1][0].item() == 1 and resposta_teste[i].item() == 0:
            falso_positivo += 1

    
    arq.write("--------------\n\n")





print("total: " + str(len(entrada_teste))+ "\n")
print("Certos: " + str(certos) + " Porcentagem: " + str(certos/len(entrada_teste))+ "\n")
print("Errados: " + str(errados) + " Porcentagem: " + str(errados/len(entrada_teste))+ "\n")
print("Falso negativo: " + str(falso_negativo) + "\n")
print("Falso positivo: " + str(falso_positivo) + "\n")
# print("Desconhecido: " + str(desconhecido) + " Porcentagem: " + str(desconhecido/len(entrada_teste))+ "\n")