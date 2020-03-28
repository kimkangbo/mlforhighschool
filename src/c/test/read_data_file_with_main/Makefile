CC = gcc
CFLAGS = -Wall
TARGET = read_data_file_with_main.exe
OBJECTS = read_data_file_with_main.o

all : $(TARGET)

$(TARGET) : $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^
	
clean :
		rm *.o $(TARGET)