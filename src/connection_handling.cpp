#include "connection_handling.hpp"

#include <iostream>

int rabbitmq_new_connection(amqp_connection_state_t  *conn, 
                            amqp_socket_t           **socket) {

    *conn   = (amqp_new_connection());
    *socket = amqp_tcp_socket_new(*conn);
    
    if (!*socket) {
        std::cout << "[daemon_d] Error creating socket" 
                  << std::endl;
        return 1;
    }

    int status = amqp_socket_open(*socket,
                                  AMQP_SERVER, 
                                  AMQP_PORT);

    if (status) {
        std::cout << "[daemon_d] Error opening socket: " 
                  << amqp_error_string2(status)
                  << std::endl;
        return 1;
    }

    amqp_rpc_reply_t login_reply = amqp_login(*conn,                    // Establishing a login connection on the specified connection state
                                              AMQP_VHOST,               // Virtual host for RabbitMQ
                                              1,                        // Channel number
                                              131072,                   // Frame max size (maximum size of frames to accept)
                                              0,                        // Heartbeat interval (0 to disable heartbeats, otherwise set in seconds)
                                              AMQP_SASL_METHOD_PLAIN,   // Authentication method (AMQP_SASL_METHOD_PLAIN for plain text) 
                                              AMQP_USER,                // RabbitMQ username for authentication
                                              AMQP_PASSWORD);           // RabbitMQ password for authentication

    if (login_reply.reply_type != AMQP_RESPONSE_NORMAL) {
        std::cout << "[daemon_d] Login failed\n" << std::endl;
        return 1;
    }
    
    // Open the channel
    amqp_channel_open(*conn, 1);
    amqp_rpc_reply_t channel_reply = amqp_get_rpc_reply(*conn);
    if (channel_reply.reply_type != AMQP_RESPONSE_NORMAL) {
        std::cout << "[daemon_d] Error opening channels\n" << std::endl;
        return 1;
    }

    return 0;
}

void send_message(amqp_connection_state_t *conn, 
                  char                    *message, 
                  char const              *queue) {

    amqp_basic_properties_t props;

    props._flags        = AMQP_BASIC_CONTENT_TYPE_FLAG      // props.content_type
                        | AMQP_BASIC_DELIVERY_MODE_FLAG;    // props.delivery_mode

    props.content_type  = amqp_cstring_bytes("text/plain");
    props.delivery_mode = 2;

    auto success = amqp_basic_publish(*conn,                        // state
                                      1,                            // channel
                                      amqp_empty_bytes,             // exchange
                                      amqp_cstring_bytes(queue),    // routing_key
                                      1,                            // mandatory
                                      0,                            // immediate
                                      &props,                       // properties
                                      amqp_cstring_bytes(message)); // body

    if (success > 0)
        std::cout << "[daemon_d] Message could not be delivered to client" << std::endl;
}

const char *receive_message(amqp_connection_state_t *conn, 
                            char const               *queue) {

    char *received_message = nullptr;

    // Declare the queue
    amqp_queue_declare(*conn,                        // state
                       1,                            // channel
                       amqp_cstring_bytes(queue),   // queue
                       1,                           // passive
                       1,                           // durable
                       1,                           // exclusive
                       0,                           // auto_delete
                       amqp_empty_table);           // arguments

    auto rcp_reply = amqp_get_rpc_reply(*conn);

    if(rcp_reply.reply_type != AMQP_RESPONSE_NORMAL)
        return nullptr;

    amqp_basic_consume(*conn,                       // state
                       1,                           // channel
                       amqp_cstring_bytes(queue),   // queue
                       amqp_empty_bytes,            // consumer_tag
                       0,                           // no_local
                       0,                           // no_ack
                       0,                           // exclusive
                       amqp_empty_table);           // arguments

    amqp_rpc_reply_t consume_reply = amqp_get_rpc_reply(*conn);

    if (consume_reply.reply_type != AMQP_RESPONSE_NORMAL) {
        std::cout << "[daemon_d] Error starting to consume messages" << std::endl;
        return nullptr;
    }
    
    amqp_rpc_reply_t res;
    amqp_envelope_t envelope;

    amqp_maybe_release_buffers(*conn);
    res = amqp_consume_message(*conn,       // state
                               &envelope,  // envelope
                               NULL,       // timeout
                               0);         // flags

    if (res.reply_type == AMQP_RESPONSE_NORMAL) {
        // Acknowledge the received message
        amqp_basic_ack(*conn, 1, envelope.delivery_tag, 0);

        received_message = new char[envelope.message.body.len + 1];  // +1 for the null terminator
        if (received_message) {
            std::memcpy(received_message, envelope.message.body.bytes, envelope.message.body.len);
            received_message[envelope.message.body.len] = '\0';
            amqp_destroy_envelope(&envelope);
        }
    } else {
        std::cout << "[daemon_d] No message received or an error occurred" << std::endl;
    }

    if (received_message)
        return received_message;

    return nullptr;
}

void close_connections(amqp_connection_state_t *conn) {
    amqp_channel_close(*conn, 1, AMQP_REPLY_SUCCESS);
    amqp_connection_close(*conn, AMQP_REPLY_SUCCESS);
    amqp_destroy_connection(*conn);
}

