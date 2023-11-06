#include "connection_handling.hpp"

int rabbitmq_new_connection(amqp_connection_state_t  *conn, 
                            amqp_socket_t           **socket, 
                            int                       SendChannel, 
                            int                       ReceiveChannel) {

    *conn   = (amqp_new_connection());
    *socket = amqp_tcp_socket_new(*conn);
    
    if (!*socket) {
        printf("Error creating socket\n");
        return 1;
    }

    int status = amqp_socket_open(*socket,
                                  AMQP_SERVER, 
                                  AMQP_PORT);

    if (status) {
        printf("Error opening socket\n");
        return 1;
    }

    amqp_rpc_reply_t login_reply = amqp_login(*conn,                    // Establishing a login connection on the specified connection state
                                              AMQP_VHOST,               // Virtual host for RabbitMQ
                                              2,                        // Channel number
                                              131072,                   // Frame max size (maximum size of frames to accept)
                                              0,                        // Heartbeat interval (0 to disable heartbeats, otherwise set in seconds)
                                              AMQP_SASL_METHOD_PLAIN,   // Authentication method (AMQP_SASL_METHOD_PLAIN for plain text) 
                                              AMQP_USER,                // RabbitMQ username for authentication
                                              AMQP_PASSWORD);           // RabbitMQ password for authentication

    if (login_reply.reply_type != AMQP_RESPONSE_NORMAL) {
        printf("Login failed\n");
        return 1;
    }
    
    // Open the channels
    amqp_channel_open(*conn, ReceiveChannel);
    amqp_channel_open(*conn, SendChannel);
    amqp_rpc_reply_t channel_reply = amqp_get_rpc_reply(*conn);
    if (channel_reply.reply_type != AMQP_RESPONSE_NORMAL) {
        printf("Error opening channels\n");
        return 1;
    }

    return 0;
}

void send_message(amqp_connection_state_t *conn, 
                  char                    *message, 
                  char const              *queue, 
                  int                      SendChannel) {

    amqp_basic_properties_t props;

    props._flags        = AMQP_BASIC_CONTENT_TYPE_FLAG      // props.content_type
                        | AMQP_BASIC_DELIVERY_MODE_FLAG;    // props.delivery_mode

    props.content_type  = amqp_cstring_bytes("text/plain");
    props.delivery_mode = 2;

    auto success = amqp_basic_publish(*conn,                        // state
                                      SendChannel,                  // channel
                                      amqp_empty_bytes,             // exchange
                                      amqp_cstring_bytes(queue),    // routing_key
                                      1,                            // mandatory
                                      0,                            // immediate
                                      &props,                       // properties
                                      amqp_cstring_bytes(message)); // body

    if (success > 0)
        printf("Message couldn't be delivered");
}

const char *receive_message(amqp_connection_state_t  *conn, 
                            char const               *queue, 
                            int                       ReceiveChannel) {

    char *received_message = nullptr;

    // Declare the queue
    amqp_queue_declare(*conn,                       // state
                       ReceiveChannel,              // channel
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
                       ReceiveChannel,              // channel
                       amqp_cstring_bytes(queue),   // queue
                       amqp_empty_bytes,            // consumer_tag
                       0,                           // no_local
                       0,                           // no_ack
                       0,                           // exclusive
                       amqp_empty_table);           // arguments

    amqp_rpc_reply_t consume_reply = amqp_get_rpc_reply(*conn);

    if (consume_reply.reply_type != AMQP_RESPONSE_NORMAL) {
        printf("Error starting to consume messages\n");
        return nullptr;
    }
    
    amqp_rpc_reply_t res;
    amqp_envelope_t envelope;

    amqp_maybe_release_buffers(*conn);
    res = amqp_consume_message(*conn,      // state
                               &envelope,  // envelope
                               NULL,       // timeout
                               0);         // flags

    if (res.reply_type == AMQP_RESPONSE_NORMAL) {
        // Acknowledge the received message
        amqp_basic_ack(*conn, ReceiveChannel, envelope.delivery_tag, 0);

        received_message = new char[envelope.message.body.len + 1];  // +1 for the null terminator
        if (received_message) {
            std::memcpy(received_message, envelope.message.body.bytes, envelope.message.body.len);
            received_message[envelope.message.body.len] = '\0';
            amqp_destroy_envelope(&envelope);
        }
    } else {
        printf("No message received or an error occurred.\n");
    }

    if (received_message)
        return received_message;

    return nullptr;
}

int start_consuming(amqp_connection_state_t *conn, 
                    char const              *ClientQueue,
                    char const              *DaemonQueue,
                    int                      ReceiveChannel,
                    int                      SendChannel) {

    // Declare the client queue
    amqp_queue_declare(*conn,
                       ReceiveChannel, 
                       amqp_cstring_bytes(ClientQueue), 
                       0, 
                       1, 
                       0, 
                       0, 
                       amqp_empty_table);

    // Declare the daemon queue
    amqp_queue_declare(*conn,
                       ReceiveChannel, 
                       amqp_cstring_bytes(DaemonQueue), 
                       0, 
                       1, 
                       0, 
                       0, 
                       amqp_empty_table);

    amqp_rpc_reply_t consume_reply = amqp_get_rpc_reply(*conn);

    if (consume_reply.reply_type != AMQP_RESPONSE_NORMAL) {
        printf("Error starting to consume messages\n");
        return 1;
    }

    printf("Waiting for messages. Press Ctrl+C to exit.\n");

    while (true) {
        auto *received_message = receive_message(conn,              // conn
                                                 DaemonQueue,       // queue
                                                 ReceiveChannel);   // SendChannel

        if (received_message) {
            printf("Received a message: %s\n", received_message); 

            const char *sent_message = "Hello client";

            send_message(conn,                  // conn
                        (char *)sent_message,   // message
                        ClientQueue,            // queue
                        SendChannel);           // SendChannel

            printf("I sent a message back to the client\n");
            delete[] received_message;
        }
    }
    
    return 0;
}

void close_connections(amqp_connection_state_t *conn, int ReceiveChannel, int SendChannel) {
    amqp_channel_close(*conn, ReceiveChannel, AMQP_REPLY_SUCCESS);
    amqp_channel_close(*conn, SendChannel, AMQP_REPLY_SUCCESS);
    amqp_connection_close(*conn, AMQP_REPLY_SUCCESS);
    amqp_destroy_connection(*conn);
}

