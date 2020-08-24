from Lib import * 
import torch
from sklearn.model_selection import train_test_split

### modelo da rede neural e parametros
model = ANN()
model.cuda()
model.train()

# criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.00055310) # 0.00015822  0.00055310
# optimizer = torch.optim.Adam(model.parameters(), lr=0.00015822) #, betas=[0.000076253698849,0.000076253698849], weight_decay=0.85565561 


# entrada, respostas_vetor = load_mfcc_GPU()
data = torch.load('data.pt')
target = torch.load('target.pt')

entrada, entrada_validacao , respostas_vetor, respostas_vetor_validacao = train_test_split(data, target, test_size=0.2, random_state=42)
entrada.requires_grad_()


# print("Começando o treino")
# running_loss = 0.0

# for epoch in range(50):  # loop over the dataset multiple times
#     print(epoch)
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

model.load_state_dict(torch.load("net-lr_diferente.pth"))
model.eval()
desconhecido = 0
certos = 0
errados = 0
resposta = 0

for i in range (len(entrada_validacao)):

    outputs = model(entrada_validacao[i].unsqueeze(0).unsqueeze(0))
    predicted = torch.max(outputs.unsqueeze(0), 1)
    # print(outputs)
    # print (torch.max(outputs.unsqueeze(0), 1))
    # input("enter")

    if predicted[1][0].item() == respostas_vetor_validacao[i].item():
        certos += 1
    else:        
        errados += 1
    





print("total: " + str(len(entrada_validacao)))
print("Certos: " + str(certos) + " Porcentagem: " + str(certos/len(entrada_validacao)))
print("Errados: " + str(errados) + " Porcentagem: " + str(errados/len(entrada_validacao)))
print("Desconhecido: " + str(desconhecido) + " Porcentagem: " + str(desconhecido/len(entrada_validacao)))