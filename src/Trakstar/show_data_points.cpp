#include "stdafx.h"


using namespace std;





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

    //system setup
    double measure_rate = stod(argv[1]);  //measurement rate in Hz (CLI)
    double max_range = stod(argv[2]);  // maximum range parameter (36, 72, 144 in). Anything above 36 will have lower resolution (CLI)
    short trans_select = 0; //selects transmitter based on id (default 0) (CLI)

    //sensor setup
    BOOL filter_wide, filter_narrow;
    if (argv[3] == "wide") {
        filter_wide = true;
        filter_narrow = false;
    } else {
        filter_wide = false;
        filter_narrow = true;
    }

    HEMISPHERE_TYPE hem;
    switch(argv[4]) {
        case "front":
            hem = FRONT;
            break;
        case "back":
            hem = BACK;
            break;
        case "top":
            hem = TOP;
            break;
        case "bottom":
            hem = BOTTOM;
            break;
        case "left":
            hem = LEFT;
            break;
        case "right":
            hem = RIGHT;
            break;
        default:
            hem = FRONT;
    }

    double azim = 0;
    double elev = 0;
    double roll = 0;

    //Transmitter settings vars
    BOOL xyz_ref_frame = 0;

    string file_naming = argv[5];

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
    Save system settings (for reference) -- SHRISHA
    */
    string str_path = "./" + file_naming + "/" + file_naming + ".ini";
    char * savePath = const_cast<char*>(str_path.c_str());
    errorCode = SaveSystemConfiguration(savePath);
    if(errorCode!=BIRD_ERROR_SUCCESS) errorHandler(errorCode);





    /*
    Data Collection -- SHRISHA
    */
    printf("Collecting Data \n");

    int client_socket = socket(AF_INET, SOCK_STREAM, 0);
    
    // Set up the server address
    struct sockaddr_in server_address;
    server_address.sin_family = AF_INET;
    server_address.sin_port = htons(12346);
    inet_pton(AF_INET, "127.0.0.1", &server_address.sin_addr);
    
    // Connect to the socket
    int connection_status = connect(client_socket, (struct sockaddr*) &server_address, sizeof(server_address));
    if (connection_status == -1) {
        cerr << "Could not connect to server" << endl;
        return 1;
    }

    char buffer[4096];
    snprintf(buffer, sizeof(buffer), "Sensor ID, Status, X (mm), Y (mm), Z (mm), Azimuth, Elevation, Roll, trakStart Time (ms), Quality");
    int bytes_sent = send(client_socket, buffer, strlen(buffer), 0);
    if (bytes_sent == -1) {
        close(client_socket);
        std::cerr << "Could not send data" << std::endl;
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
                string time_str = to_string(record[sensorID].time); //to format trakSTAR time into string so it goes in properly
                snprintf(buffer, sizeof(buffer), "%u, %u, %f, %f, %f, %f, %f, %f, %s, %u", sensorID, status, record[sensorID].x, record[sensorID].y, record[sensorID].z, record[sensorID].a, record[sensorID].e, record[sensorID].r, time_str, record[sensorID].quality);
                int bytes_sent = send(client_socket, buffer, strlen(buffer), 0);
                if (bytes_sent == -1) {
                    close(client_socket);
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

}