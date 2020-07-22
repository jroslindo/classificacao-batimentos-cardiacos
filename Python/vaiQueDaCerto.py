import matlab.engine 
import psutil, os #utilizado para colocar o processo em alta prioridade
import sys

def elevate_process_priority ():
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)

def call_matlab_script(file_name):
    eng = matlab.engine.start_matlab()
    eng.cd(r'..\\Matlab_segmentacao')
    resultado = eng.joao_script(file_name,nargout=1)
    variavel = resultado
    eng.quit()

    print(variavel)

def main(argumento):
    # print(argumento)
    elevate_process_priority()
    call_matlab_script(argumento)

main(sys.argv[1])