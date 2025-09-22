#include "stdafx.h"


using namespace std;

#define DEFAULT_BUFLEN 128

/*Standard error handler. Whenever some settings are changed,
the error handler is run in ordder to check that everything is going
correctly. If not, it prints what is wrong and exits the program so as not 
to messs anything up  -- SHRISHA */
void errorHandler(int error){
    char    buffer[1024];
	char    *pBuffer = &buffer[0];
	int     numberBytes;

	while(error!=BIRD_ERROR_SUCCESS)
	{
		error = GetErrorText(error, pBuffer, sizeof(buffer), SIMPLE_MESSAGE);
		numberBytes = strlen(buffer);
		buffer[numberBytes] = '\n';		// append a newline to buffer
		printf("%s", buffer);
	}

    sleep(10);
	exit(0);
}


int main(int argc, char* argv[])  {

    //The following are used for setup, data collection, and logging of current system settings / info
    CSystem     ATC3DG;		    // a pointer to a single instance of the system class	
	CSensor     *pSensor;		// a pointer to an array of sensor objects
    CXmtr		*pXmtr;			// a pointer to an array of transmitter objects

    int         i;              // used in general to loop through
    int         errorCode;      // for error handling

    printf("Configuring system settings\n");
    //system setup
    double measure_rate = 90;  //measurement rate in Hz (CLI)
    double max_range = 36;  // maximum range parameter (36, 72, 144 in). Anything above 36 will have lower resolution (CLI)
    short trans_select = 0; //selects transmitter based on id (default 0) (CLI)
    BOOL filter_wide = true; 
    BOOL filter_narrow = false;



    HEMISPHERE_TYPE hem = FRONT; //default hemisphere is front (CLI)

    double azim = 0;
    double elev = 0;
    double roll = 0;

    //Transmitter settings vars
    BOOL xyz_ref_frame = 0;


    /*
    System Initialize -- JOYCE AND SHRISHA
    */
    errorCode = InitializeBIRDSystem();
    if (errorCode != BIRD_ERROR_SUCCESS) errorHandler(errorCode);

    //Get system data so we can instatiate the system class
    errorCode = GetBIRDSystemConfiguration(&ATC3DG.m_config);
	if(errorCode!=BIRD_ERROR_SUCCESS) errorHandler(errorCode);

    //Get sensor data so we can fill out the system class to more depth, as well as sensor class. Lets us know 
    //which sensors are connected so we can dynamically loop through only the connected sensors to collect data
    pSensor = new CSensor[ATC3DG.m_config.numberSensors];
	for(i=0;i<ATC3DG.m_config.numberSensors;i++)
	{
		errorCode = GetSensorConfiguration(i, &pSensor[i].m_config);
		if(errorCode!=BIRD_ERROR_SUCCESS) errorHandler(errorCode);
	}

    //Gets transmitter data to help with (a) finding it and (b) turning it on, since we are only ever using one
    pXmtr = new CXmtr[ATC3DG.m_config.numberTransmitters];
	for(i=0;i<ATC3DG.m_config.numberTransmitters;i++)
	{
		errorCode = GetTransmitterConfiguration(i, &pXmtr[i].m_config);
		if(errorCode!=BIRD_ERROR_SUCCESS) errorHandler(errorCode);
	}



    /*
    System settings setup (basic settings) -- SHRISHA 
    */

    SET_SYSTEM_PARAMETER(SELECT_TRANSMITTER, trans_select);
	SET_SYSTEM_PARAMETER(MEASUREMENT_RATE, measure_rate);
    SET_SYSTEM_PARAMETER(MAXIMUM_RANGE, max_range);
    SET_SYSTEM_PARAMETER(METRIC, true);



    /*
    Sensor settings setup -- SHRISHA
    */

    USHORT  sensorID;

    for(sensorID = 0; sensorID<ATC3DG.m_config.numberSensors; sensorID++) 
    {
        SET_SENSOR_PARAMETER(sensorID, DATA_FORMAT, DOUBLE_POSITION_ANGLES_TIME_Q); //data type (xyz, are, time)
        {
            DOUBLE_ANGLES_RECORD anglesRecord = {azim, elev, roll};  //can account for Azim, Elev, and Roll respectivley
            SET_SENSOR_PARAMETER(sensorID, ANGLE_ALIGN, anglesRecord);
        }
        /* FRONT, BACK, TOP, BOTTOM, LEFT, or RIGHT. Determines the +/- of the measurements. The magnitude will always be 
        right but +/- needs to be accounted for. FRONT is default, read documentaiton on how to account for +/- */
        SET_SENSOR_PARAMETER(sensorID, HEMISPHERE, hem);  
        /*To chose between Wide and Norrow, Wide eliminates nose between 32 - 72 Hz at the cost of higher delay between 
        measurement and data output. To Decrease this time delay,  set Wide to False and Narrow to True (narrow detects lower range*/
        SET_SENSOR_PARAMETER(sensorID, FILTER_AC_WIDE_NOTCH, filter_wide); 
        SET_SENSOR_PARAMETER(sensorID, FILTER_AC_NARROW_NOTCH, filter_narrow); 
    }




    /*
    Transmitter settings setup -- JOYCE
    */

    USHORT transmitterID = 0;

    {
		DOUBLE_ANGLES_RECORD anglesRecord = {0, 0, 0}; //structure of angles
		SET_TRANSMITTER_PARAMETER(transmitterID, REFERENCE_FRAME, anglesRecord);
	}
	SET_TRANSMITTER_PARAMETER(transmitterID, XYZ_REFERENCE_FRAME, xyz_ref_frame);  //sets the angle reference frame with respect to transmitter tilt. False means not with respect to. 









    /*
    Data Collection -- SHRISHA
    */
    printf("Collecting Data \n");

    WSADATA wsa_data;
    int result = WSAStartup(MAKEWORD(2,2), &wsa_data);
    if (result != 0) {
        cerr << "WSA startup failed: " << result << endl;
        return 1;
    }



    SOCKET client_socket = socket(AF_INET, SOCK_STREAM, 0);

    if (client_socket == INVALID_SOCKET) {
        cerr << "Error creating socket: " << WSAGetLastError() << endl;
        WSACleanup();
        return 1;
    }
    
    // Set up the server address
    struct sockaddr_in server_address;
    server_address.sin_family = AF_INET;
    server_address.sin_port = htons(12346);
    server_address.sin_addr.s_addr = inet_addr("127.0.0.1");

    // // Connect to the socket
    // int connection_status = connect(client_socket, (struct sockaddr*) &server_address, sizeof(server_address));
    // if (connection_status == -1) {
    //     cerr << "Could not connect to server" << endl;
    //     return 1;
    // }

        // Connect to the server
    if (connect(client_socket, reinterpret_cast<sockaddr*>(&server_address), sizeof(server_address)) == SOCKET_ERROR) {
        cerr << "Could not connect to server: " << WSAGetLastError() << endl;
        closesocket(client_socket);
        WSACleanup();
        return 1;
    }

    DOUBLE_POSITION_ANGLES_TIME_Q_RECORD record[8*4];
    DOUBLE_POSITION_ANGLES_TIME_Q_RECORD *pRecord = record;


    while(1) {
		errorCode = GetSynchronousRecord(ALL_SENSORS, pRecord, sizeof(record)[0] * ATC3DG.m_config.numberSensors);
		if(errorCode!=BIRD_ERROR_SUCCESS) errorHandler(errorCode);

		// scan the sensors and request a record if the sensor is physically attached
		for(sensorID=0; sensorID < ATC3DG.m_config.numberSensors; sensorID++)
		{
			// get the status of the last data record and only report the data if everything is okay
			unsigned int status = GetSensorStatus(sensorID);

			if (status == VALID_STATUS)
			{
                                

                char tempBuffer[1];  // Just a temporary buffer with size 1
                int bytesNeeded;

                // Calculate the size without writing to the buffer
                sprintf(tempBuffer, "%u,%f,%f,%f,%f,%f,%f,%f%n", sensorID, record[sensorID].x, record[sensorID].y, record[sensorID].z, record[sensorID].a, record[sensorID].e, record[sensorID].r, record[sensorID].time, &bytesNeeded);




                char buffer[128];
                int bytesWritten;
                memset(buffer, 0, 128); //possible source of error - filling char with all 0? Maybe increase value of memory?               
                bytesWritten = sprintf(buffer, "%d,%u,%f,%f,%f,%f,%f,%f,%f", bytesNeeded, sensorID, record[sensorID].x, record[sensorID].y, record[sensorID].z, record[sensorID].a, record[sensorID].e, record[sensorID].r, record[sensorID].time);

                printf("bytesWritten: %d\n", bytesWritten);


                cout << buffer << endl;
                int bytes_sent = send(client_socket, buffer, strlen(buffer), 0);
                printf("Bytes_sent: %d\n", bytes_sent);


                // Receive acknowledgment from the server
                char acknowledgment_buffer[DEFAULT_BUFLEN];
                int bytes_received = recv(client_socket, acknowledgment_buffer, DEFAULT_BUFLEN, 0);

                if (bytes_received == SOCKET_ERROR) {
                    cerr << "Error receiving acknowledgment: " << WSAGetLastError() << endl;
                    closesocket(client_socket);
                    WSACleanup();
                    return 1;
                }

                acknowledgment_buffer[bytes_received] = '\0';  // Null-terminate the received data
                cout << "Received acknowledgment: " << acknowledgment_buffer << endl;


                if (bytes_sent == SOCKET_ERROR) {
                    cerr << "Error sending data: " << WSAGetLastError() << endl;
                    closesocket(client_socket);
                    WSACleanup();
                    return 1;
                }

                if (bytes_sent < 0) {
                    printf("Error Bytes_sent: %d\n", bytes_sent);
                    sleep(1);
                    closesocket(client_socket);
                    USHORT id = -1;
                    printf("Socket connection stopped. Shutting Down");
                    errorCode = SetSystemParameter(SELECT_TRANSMITTER, &id, sizeof(id));
                    if(errorCode!=BIRD_ERROR_SUCCESS) errorHandler(errorCode);
                    return 1;
                }
                //For reference: myFile << sensorID << "," << status << "," <<  record[sensorID].x  << "," <<  record[sensorID].y << "," <<  record[sensorID].z << "," <<  record[sensorID].a << "," <<  record[sensorID].e << "," <<  record[sensorID].r << "," <<  time_str << "," <<  record[sensorID].quality << endl;
			}  
		}

	}

        // Cleanup
    closesocket(client_socket);
    WSACleanup();

}