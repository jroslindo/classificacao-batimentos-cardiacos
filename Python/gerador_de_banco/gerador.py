import sys
import os
import threading

global numero_de_threads_finalizadas
numero_de_threads_finalizadas = 0
global semaforo
semaforo = threading.Semaphore()


def THREAD(lista):
    print("iniciando thread")
    os.system("python ..\\segmentacao\\joao_segmentacao.py " + str(lista))
    os.system("python ..\\mfcc\\joao_mfcc.py " + str(lista)) 

    global semaforo
    global numero_de_threads_finalizadas

    semaforo.acquire()

    numero_de_threads_finalizadas +=1
    print(numero_de_threads_finalizadas)

    semaforo.release()


def main (min,max):
    os.system("cls")
    print("---Iniciando")
    lista = os.listdir("..\..\Banco_A")
    
    if max > len(lista):
        print("maior")
        exit()

    
    l=min

    while l<max:
        print("estou em " + str(l))
        j=l

        threads = []
        while j<l+10:
            try:
                t = threading.Thread(target=THREAD, args=(lista[j],))
                threads.append(t)
            except:
                print("erro")
                        
            
            j+=1
        
        for i in threads:
            i.start()
        
        for i in threads:
            i.join()
        
        l=j

    

    print("---Fim")

    # os.system("python ..\\segmentacao\\joao_segmentacao.py" + )


main(int(sys.argv[1]),int(sys.argv[2]))