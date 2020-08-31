from Lib import * 
import torch
from sklearn.model_selection import train_test_split

### modelo da rede neural e parametros
model = ANN()
model.cuda()
model.train()


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.00055310) # 0.00015822  0.00055310
data = torch.load('data.pt')
target = torch.load('target.pt')
entrada, entrada_validacao , respostas_vetor, respostas_vetor_validacao = train_test_split(data, target, test_size=0.2, random_state=42)
entrada.requires_grad_()



# print("Começando o treino")
# running_loss = 0.0

# for epoch in range(50):  # loop over the dataset multiple times
#     print("epoca " + str(epoch))
#     for i in range(len(entrada)):
#         entrada_aux = entrada[i].unsqueeze(0)

#         optimizer.zero_grad()

#         outputs = model(entrada_aux.unsqueeze(0))

#         # if i == 0:
#         #     print(outputs)
#         #     print(respostas_vetor[i])
#         #     print("\n--------------------------\n")

#         # loss = criterion(outputs, respostas_vetor[i].unsqueeze(0))
#         loss = criterion(outputs.unsqueeze(0), respostas_vetor[i].unsqueeze(0)) #respostas_vetor[i].unsqueeze(0)

#         if i == 0:
#             loss.backward(retain_graph=True)
#         else:
#             loss.backward(retain_graph=True)

#         optimizer.step()

#         running_loss += loss.item()

#     if (epoch % 10 == 9):
#         print(str(running_loss/(len(entrada)*epoch)) + "\n---------------------\n")



# print('Finished Training:  ' + str(running_loss/(len(entrada)*epoch)) + " --Salvando o modelo")

# torch.save(model.state_dict(), "net.pth")













print("validando: \n\n")

model.load_state_dict(torch.load("net.pth"))
model.eval()
desconhecido = 0
certos = 0
errados = 0
resposta = 0

arq = open("log.txt", "w")

for i in range (len(entrada_validacao)):

    outputs = model(entrada_validacao[i].unsqueeze(0).unsqueeze(0))
    predicted = torch.max(outputs.unsqueeze(0), 1)
    arq.write(str(outputs) + "\n")
    # print (torch.max(outputs.unsqueeze(0), 1))
    arq.write("previsto: " + str(predicted[1][0].item()) + "\n")
    arq.write("resposta: " + str(respostas_vetor_validacao[i].item())+ "\n")
    # input("enter")

    if predicted[1][0].item() == respostas_vetor_validacao[i].item():
        arq.write("certo"+ "\n")
        certos += 1
    else:        
        arq.write("errado"+ "\n")
        errados += 1
    
    arq.write("--------------\n\n")





arq.write("total: " + str(len(entrada_validacao))+ "\n")
arq.write("Certos: " + str(certos) + " Porcentagem: " + str(certos/len(entrada_validacao))+ "\n")
arq.write("Errados: " + str(errados) + " Porcentagem: " + str(errados/len(entrada_validacao))+ "\n")
arq.write("Desconhecido: " + str(desconhecido) + " Porcentagem: " + str(desconhecido/len(entrada_validacao))+ "\n")