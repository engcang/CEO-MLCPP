#include "ceo_mlcpp_flight_main.h"
#include <signal.h>
void signal_handler(sig_atomic_t s) {
  std::cout << "You pressed Ctrl + C, exiting" << std::endl;
  exit(1);
}




int main(int argc, char **argv){

    signal(SIGINT, signal_handler); // to exit program when ctrl+c

    ros::init(argc, argv, "ceo_mlcpp_class_node");
    ros::NodeHandle n("~");
    ceo_mlcpp_flight_class ceo_mlcpp_flight_(n);

    ros::AsyncSpinner spinner(6);
    spinner.start();
    ros::waitForShutdown();

    return 0;
}