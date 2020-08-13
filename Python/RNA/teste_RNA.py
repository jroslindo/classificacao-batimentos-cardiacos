from Lib import * 

net = ANN()
net.cuda()
net.load_state_dict(torch.load("net.pth"))


with open("c0001.txt", "rb") as fp:
    retorno = torch.cuda.FloatTensor(pickle.load(fp))

outputs = net(retorno.unsqueeze(0).unsqueeze(0))

print(outputs)