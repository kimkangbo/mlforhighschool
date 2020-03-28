CC = gcc
CFLAGS = -Wall
TARGET = mnist_test.exe
OBJECTS = img_mnist.o nn.o main.o

all : $(TARGET)

$(TARGET) : $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^
	
clean :
		rm *.o $(TARGET)