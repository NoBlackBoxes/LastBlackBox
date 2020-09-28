/*  Code to run a simple client in the internet domain communication via
    TCP.
    Peter Vincent
    Adapted from http://www.cs.rpi.edu/~moorthy/Courses/os98/Pgms/socket.html
*/

#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <stdbool.h>

void error(const char *msg)
{
    perror(msg);
    exit(1);
}

int main(int argc, char *argv[])
{
    int sockfd, portno = 52596, n;
    struct sockaddr_in serv_addr;
    struct hostent *server;

    char buffer[256];
    if (argc < 2){
        fprintf(stderr,"usage %s hostname, portnum \n", argv[0]);
        exit(1);
    }
    portno = atoi(argv[2]);
    sockfd = socket(AF_INET, SOCK_STREAM,0);
    if (sockfd < 0) error((const char*) "ERROR opening socket");
    server = gethostbyname(argv[1]);
    if (server == NULL) {
        fprintf(stderr,"Host not found, check arguments\n");
        exit(1);
    }
    memset((char*)&serv_addr,'\0',sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char*)server -> h_addr_list[0],(char*)&serv_addr.sin_addr.s_addr,
    server->h_length);
    serv_addr.sin_port = htons(portno);

    if (connect(sockfd,(struct sockaddr*)&serv_addr,sizeof(serv_addr)) < 0) error((const char *) "ERROR connecting");
    printf("Enter command now \n");
    memset(buffer,'\0',256);
    fgets(buffer,255,stdin);
    n = write(sockfd,buffer,strlen(buffer));
    if (n < 0) error((const char*)"ERROR writing to socket");
    memset(buffer,'\0',256);
    n = read(sockfd,buffer,255);
    if (n < 0) error((const char*)"ERROR reading from socket");
    fprintf(stdout,"%s\n",buffer);
    return 0;
}