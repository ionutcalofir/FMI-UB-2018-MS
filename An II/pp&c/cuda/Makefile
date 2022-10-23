createData: createData.cpp
	g++ -std=c++17 -o createData createData.cpp

testData: createData
	./createData

cleanData:
	rm -f createData image.bin image_height.bin image_width.bin image_channels.bin kernel.bin kernel_height.bin kernel_width.bin kernel_channels.bin

createMain: main.cpp
	nvcc -g -c main.cpp
	nvcc -c conv.cu
	nvcc -c max_pool.cu
	nvcc -c fc.cu
	nvcc main.o conv.o max_pool.o fc.o -o main

testMain: main
	./main

cleanMain:
	rm -f main main.o conv.o
