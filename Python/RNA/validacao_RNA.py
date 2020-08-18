from Lib import * 

net = ANN()
net.cuda()
net.load_state_dict(torch.load("net.pth"))
dados,respostas = load_mfcc_GPU_validacao()


desconhecido = 0
certos = 0
errados = 0
resposta = 0

for i in range (len(dados)):

    outputs = net(dados[i].unsqueeze(0).unsqueeze(0))
    predicted = torch.max(outputs.unsqueeze(0), 1)

    if predicted[1][0].item() == respostas[i].item():
        certos += 1
    else:
        errados += 1
    





print("total: " + str(len(dados)))
print("Certos: " + str(certos) + " Porcentagem: " + str(certos/len(dados)))
print("Errados: " + str(errados) + " Porcentagem: " + str(errados/len(dados)))
print("Desconhecido: " + str(desconhecido) + " Porcentagem: " + str(desconhecido/len(dados)))