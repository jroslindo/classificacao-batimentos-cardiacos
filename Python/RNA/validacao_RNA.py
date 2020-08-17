from Lib import * 

net = ANN()
net.cuda()
net.load_state_dict(torch.load("net.pth"))
dados,respostas = load_mfcc_GPU_validacao()


desconhecido = 0
certos = 0
errados = 0

i = 0
while i < len(dados):

    outputs = net(dados[i].unsqueeze(0).unsqueeze(0))

    if outputs >= 0.70:

        outputs = 1
        if outputs == respostas[i]:
            certos += 1
        else:
            errados += 1

    elif outputs <= 0.30:
        outputs = 0
        if outputs == respostas[i]:
            certos += 1
        else:
            errados += 1

    else:
        desconhecido += 1

    i += 1


print("total: " + str(len(dados)))
print("Certos: " + str(certos) + " Porcentagem: " + str(certos/len(dados)))
print("Errados: " + str(errados) + " Porcentagem: " + str(errados/len(dados)))
print("Desconhecido: " + str(desconhecido) + " Porcentagem: " + str(desconhecido/len(dados)))