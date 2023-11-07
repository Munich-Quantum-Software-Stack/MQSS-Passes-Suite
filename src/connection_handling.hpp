#pragma once

#include "rabbitmq-c/amqp.h"
#include "rabbitmq-c/tcp_socket.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstring>

// Define the RabbitMQ server connection information
#define AMQP_SERVER   "localhost"
#define AMQP_PORT     5672
#define AMQP_USER     "guest"
#define AMQP_PASSWORD "guest"
#define AMQP_VHOST    "/"
#define RESPONSESIZE  150

int         rabbitmq_new_connection(amqp_connection_state_t  *conn, 
                                    amqp_socket_t           **socket);

void        send_message(amqp_connection_state_t *conn, 
                         char                    *message, 
                         char const              *queue);

const char *receive_message(amqp_connection_state_t *conn, 
                            char const              *queue);

void        close_connections(amqp_connection_state_t *conn);

