import matlab.engine 
import psutil, os #utilizado para colocar o processo em alta prioridade
import sys
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt






def elevate_process_priority ():
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)

def call_matlab_script(file_name):
    eng = matlab.engine.start_matlab()
    eng.cd(r'..\\..\\Matlab_segmentacao') #redireciona para a pasta matlab desse projeto
    resultado = eng.joao_script(file_name,nargout=1) #chama meu script do matlab
    eng.quit()

    # arquivo = open("resultado.txt", "w") #resultado da segmentacao
    # for i in resultado:
    #     arquivo.write(str(i[0])+"\n")
    # arquivo.close()
    return resultado

def segmentation(file_name, segmentos):
    (rate,sig) = wav.read("..\\..\\Banco_A\\"+file_name)

    comecou = False
    aux = 'x'
    inicial = 'x'
    final = 'x'
    estagio = 0

    for i in range(0, len(sig)):
        anterior = aux
        aux = segmentos[i][0]

        if aux == 1 and comecou == False:            
            comecou = True
            inicial = i
        elif aux == 1 and anterior == 4 and comecou == True:
            estagio +=1
            if estagio == 3: #quando finalizar a terceira fase, finaliza o processo
                comecou = False
                final = i
                break
    
    resultado_segmentado = sig[inicial:final]
    

    wav.write("resultado.wav", rate, resultado_segmentado)

    # fig, (ax1, ax2) = plt.subplots(2)
    # ax1.plot(sig)
    # ax2.plot(resultado_segmentado)
    # plt.show()

def main(argumento):
    elevate_process_priority()
    segmentos_matlab = call_matlab_script(argumento)
    
    segmentation(argumento, segmentos_matlab)
    print("fim")
    
    

main(sys.argv[1])