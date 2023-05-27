nvcc -lopencv_world460 -lopencv_world460d -I C:\Users\hktbl\Desktop\CUDA\opencv\include -L C:\Users\hktbl\Desktop\CUDA\opencv\lib ./main.cu -o hello_world
hello_world ./image/Test.mp4 -nightVision >> log.txt
pause;