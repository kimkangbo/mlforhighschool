CC = gcc
CFLAGS = -Wall
TARGET = mnist_test.exe
OBJECTS = read_data_file.o neural_networks_structure.o main.o

all : $(TARGET)

$(TARGET) : $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^
	
clean :
		rm *.o $(TARGET)