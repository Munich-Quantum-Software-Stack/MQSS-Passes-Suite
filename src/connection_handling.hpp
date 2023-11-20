/**
 * @file connection_handling.hpp
 * @brief TODO
 * @todo Comment this source code
 */

#pragma once

#include "rabbitmq-c/amqp.h"
#include "rabbitmq-c/tcp_socket.h"
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Define the RabbitMQ server connection information
#define AMQP_SERVER "rabbitmq" //"localhost"
#define AMQP_PORT 5672
#define AMQP_USER "guest"
#define AMQP_PASSWORD "guest"
#define AMQP_VHOST "/"
#define RESPONSESIZE 150

int rabbitmq_new_connection(amqp_connection_state_t *conn,
                            amqp_socket_t **socket);

void send_message(amqp_connection_state_t *conn, char *message,
                  char const *queue);

const char *receive_message(amqp_connection_state_t *conn, char const *queue);

void close_connections(amqp_connection_state_t *conn);
