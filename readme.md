### MIDAS V1 - Data Collection System - Robotic Surgery - UVA DSA


#### Instructions

Pre-requisites
1. Alienware Laptop logged in as Student user.
2. HDMI Capture card connected via USB.
3. Trakstar Device connected via USB.

## Compiling Trakstar
1. Go to src/Trakstar directory.
2. `g++ -c show_data_points.cpp -o main.o; g++ -o main.exe main.o -L. -lATC3DG64 -lws2_32 -liconv;`
## Running the system

1. Activate Conda Environment `midasv1` using command prompt.
2. Execute `python app.py` in `src` directory.
3. Enter details as instructed in the browser window.
4. Open a different terminal and cd into `src/Trackstar` folder.
5. Execute `main.exe`.