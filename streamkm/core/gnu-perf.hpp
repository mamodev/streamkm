#pragma once
#include <iostream>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <string>

namespace streamkm {
class GnuPerfManager {
 
    // control and ack fifo from perf
    int ctl_fd = -1;
    int ack_fd = -1;
 
    // if perf is enabled
    bool enable = false;

    bool running = true;
 
    // commands and acks to/from perf
    static constexpr const char* enable_cmd = "enable";
    static constexpr const char* disable_cmd = "disable";
    static constexpr const char* ack_cmd = "ack\n";
 
    // send command to perf via fifo and confirm ack
    void send_command(const char* command) {
        if (enable) {
            write(ctl_fd, command, strlen(command));
            char ack[5];
            read(ack_fd, ack, 5);
            assert(strcmp(ack, ack_cmd) == 0);
        }
    }
 
  public:
 
    GnuPerfManager() {
        // setup fifo file descriptors
        char* ctl_fd_env = std::getenv("PERF_CTL_FD");
        char* ack_fd_env = std::getenv("PERF_ACK_FD");
        if (ctl_fd_env && ack_fd_env) {
            enable = true;
            ctl_fd = std::stoi(ctl_fd_env);
            ack_fd = std::stoi(ack_fd_env);
        } else {
            std::cout << "[WARNING] GnuPerfManager: "
                      << "PERF_CTL_FD and PERF_ACK_FD not set, "
                      << "perf will not be enabled." << std::endl;
        }
    }
 
    // public apis
 
    void pause() {
        if (running) {
            send_command(disable_cmd);
            running = false;
        }
    }
 
    void resume() {
        if (!running) {
            send_command(enable_cmd);
            running = true;
        }
    }
 
};
} // namespace streamkm