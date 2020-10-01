/*  Code to run a simple server on the Pi in the internet domain 
    TCP.
    Peter Vincent
    Adapted from http://www.cs.rpi.edu/~moorthy/Courses/os98/Pgms/socket.html

    The server will wait for a connection request from a client.
    In this code it assumes that the client will send chars.
    If the server recieves -1, it closes the socket with the client.
    If the server recieves -2, the program exits.
*/

#include <stdio.h>
#include <stdbool.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>

#define bzero(b,len) (memset((b), '\0', (len)), (void) 0)

void make_comm(int,int);
void error ( const char *msg ){
    perror ( msg );
    exit(1);
}

void dostuff (int sock)
{
   int n;
   char buffer[256];
   bzero(buffer,256);
   n = read(sock,buffer,255);
   if (n < 0) error("ERROR reading from socket");
   printf("Here is the message: %s\n",buffer);
   n = write(sock,"I got your message",18);
   if (n < 0) error("ERROR writing to socket");
}


int main(int argc, char *argv[]){
    int sockfd, newsockfd, portno, clilen, n, pid;
    char buffer[256];

    if (argc < 2){
        fprintf(stderr,"Usage portnumber\n");
        exit(1);
    }

    portno = atoi(argv[1]);
    struct sockaddr_in serv_addr, cli_addr; // Definition comes from <netinet/in.h>
    int data;
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    /*
        Three arguments for socket - 
            AF_INET - Address domain of the socket (Internet not unix)
            SOCK_STREAM - The type of socket.  We could ask to be sending chunks of data
            (and maybe this is what you want) but here we specify that we want to be 
            streaming data across the socket
            0 - THis is the protocol, and should be 0 unless you really know what you're doing
            If we specify stream sockets it will choose TCP, if we specify datagram sockets
            it will use UDP.

            If this call fails it will return -1 (see error below)
    */
    if (sockfd < 0){
        error( (const char *)"Error opening socket" );
    }
    // Initialises a buffer of 0s.  Could also use memset.
    bzero((char *) &serv_addr, sizeof(serv_addr));
     
    // Set parameters of the server
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port   = htons(portno);
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    if (bind(sockfd, (struct sockaddr*) &serv_addr, sizeof(serv_addr))<0){
        error( (const char *)"ERROR on binding");
    }
    /*
        Bind system call binds the socket we have to an address.  
        Three arguments - 
            socket file descriptor
            address it should be bound to - we pass in a reference to a structure of
            type sockaddr_in which is case to a pointer to a structure of type sockaddr
            size of the address to which it is bound
    */
    listen(sockfd,5); // 5 here specifies how many connections can be waiting whilst we process one
    clilen = sizeof(cli_addr);
    while (1) {
        newsockfd = accept(sockfd, (struct sockaddr *) &cli_addr, (socklen_t*) &clilen);
        if (newsockfd < 0) error((const char*)"Error on accepting connection");
        pid = fork();
        if (pid < 0)
                 error("ERROR on fork");
        if (pid == 0)  {
            close(sockfd);
            dostuff(newsockfd);
            exit(0);
        }
        else close(newsockfd);
    }
    return 0;
}



