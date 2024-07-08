Installation

1. Install the python requirements (requirements.txt)
2. Install Intel Realsense SDK
3. Install ZED SDK, along with CUDA 12.1; Run depth viewer software with neural and neural+ depth mode to complete optimizaiton
4. Install Zed python bindings (install_zed_python.py)


Compiling Trakstar

1. Go to src/Trakstar directory.
2. `g++ -c show_data_points.cpp -o main.o; g++ -o main.exe main.o -L. -lATC3DG64 -lws2_32 -liconv;`


OBS Web Socket Controller

1. Enable OBS websocket (Tools->WebSocket Server Settings->Enable Websocket Server)
2. Enter the OBS password and port according to the values stored in "config.py" file