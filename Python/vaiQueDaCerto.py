import matlab.engine
eng = matlab.engine.start_matlab()
eng.cd(r'..\\Matlab_segmentacao')
eng.joao_script(nargout=1)

eng.quit()