cmake_minimum_required(VERSION 3.5)
project(gsl_evaluation)

find_package(ament_cmake REQUIRED)

install(DIRECTORY .
DESTINATION share/${PROJECT_NAME}
)

ament_package()
