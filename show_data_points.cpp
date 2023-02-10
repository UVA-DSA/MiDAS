#include "stdafx.h"

using namespace std;
using json = nlohmann::json;




/*Standard error handler. Whenever some settings are changed,
the error handler is run in ordder to check that everything is going
correctly. If not, it prints what is wrong and exits the program so as not 
to messs anything up  -- SHRISHA */
void errorHandler(int error){
    char    buffer[1024];
	char    *pBuffer = &buffer[0];
	int     numberBytes;

    ofstream errorFile;
    errorFile.open("errorFile.txt");
    errorFile << "error";
    errorFile.close();

	while(error!=BIRD_ERROR_SUCCESS)
	{
		error = GetErrorText(error, pBuffer, sizeof(buffer), SIMPLE_MESSAGE);
		numberBytes = strlen(buffer);
		buffer[numberBytes] = '\n';		// append a newline to buffer
		printf("%s", buffer);
	}


    printf("\nClosing in 10 seconds\n");

    sleep(10);
	exit(0);
}


int main() {

    //The following are used for setup, data collection, and logging of current system settings / info
    CSystem     ATC3DG;		    // a pointer to a single instance of the system class	
	CSensor     *pSensor;		// a pointer to an array of sensor objects
    CXmtr		*pXmtr;			// a pointer to an array of transmitter objects

    int         i;              // used in general to loop through
    int         errorCode;      // for error handling



    /*
    Openning up JSON File and Reading Data -- SHRISHA
    */
    ifstream json_file_read;
    json_file_read.open("data.json");
    json j;
    json_file_read >> j;


    string file_naming = j.value("file_naming", "Unable_to_parse_info");
    char * name_conv = const_cast<char*>(file_naming.c_str()); // turn into char *

    //system setup
    double measure_rate = j.value("rate", 90.0);  //measurement rate in Hz (CLI)
    double max_range = j.value("range", 36.0);  // maximum range parameter (36, 72, 144 in). Anything above 36 will have lower resolution (CLI)
    short trans_select = 0; //selects transmitter based on id (default 0) (CLI)

    //sensor setup
    BOOL filter_wide, filter_narrow;
    if (j.value("filter_ac_wide_notch", 1) == 1) {
        filter_wide = true;
        filter_narrow = false;
    } else {
        filter_wide = false;
        filter_narrow = true;
    }

    HEMISPHERE_TYPE hem;
    switch(j.value("hemisphere", 1)) {
        case 0:
            hem = FRONT;
            break;
        case 1:
            hem = BACK;
            break;
        case 2:
            hem = TOP;
            break;
        case 3:
            hem = BOTTOM;
            break;
        case 4:
            hem = LEFT;
            break;
        case 5:
            hem = RIGHT;
            break;
        default:
            hem = FRONT;
    }

    double azim = j.value("azimuth", 0);
    double elev = j.value("elevation", 0);
    double roll = j.value("roll", 0);

    //Transmitter settings vars
    BOOL xyz_ref_frame = 0;


    json_file_read.close();

    int rm_status = remove("data.json");
    if (rm_status == 0) {
        printf("JSON File Deleted\n");
    }

    /*
    System Initialize -- JOYCE AND SHRISHA
    */
    printf("\n\nWelcome to DATA COLLECTION APP \n");
    printf("Initializing System\n");
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
    printf("System Initialization Successful\n");





    /*
    System settings setup (basic settings) -- SHRISHA 
    */
    printf("Starting System Settings\n");

    SET_SYSTEM_PARAMETER(SELECT_TRANSMITTER, trans_select);
	SET_SYSTEM_PARAMETER(MEASUREMENT_RATE, measure_rate);
    SET_SYSTEM_PARAMETER(MAXIMUM_RANGE, max_range);
    SET_SYSTEM_PARAMETER(METRIC, true);
    printf("System Settings Successful\n");




    /*
    Sensor settings setup -- SHRISHA
    */
    printf("Starting Sensor Settings \n");

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
    printf("Sensor Settings Successful\n");




    /*
    Transmitter settings setup -- JOYCE
    */
    printf("Starting Transmitter Settings\n");

    USHORT transmitterID = 0;

    {
		DOUBLE_ANGLES_RECORD anglesRecord = {0, 0, 0}; //structure of angles
		SET_TRANSMITTER_PARAMETER(transmitterID, REFERENCE_FRAME, anglesRecord);
	}
	SET_TRANSMITTER_PARAMETER(transmitterID, XYZ_REFERENCE_FRAME, xyz_ref_frame);  //sets the angle reference frame with respect to transmitter tilt. False means not with respect to. 
    printf("Transmitter Settings Successful\n");




    /*
    Save system settings (for reference) -- SHRISHA
    */
    string str_path = "./" + file_naming + "/" + file_naming + ".ini";
    char * savePath = const_cast<char*>(str_path.c_str());
    printf("Saving system configuration to %s\n", savePath);
    errorCode = SaveSystemConfiguration(savePath);
    if(errorCode!=BIRD_ERROR_SUCCESS) errorHandler(errorCode);





    /*
    Data Collection -- SHRISHA
    */
    printf("Collecting Data \n");

    ofstream myFile;
    myFile.open("./" + file_naming + "/" + file_naming + ".csv");
    myFile << "Sensor ID" << "," << "Status" << "," <<  "X (mm)"  << "," <<  "Y (mm)" << "," <<  "Z (mm)" << "," <<  "Azimuth" << "," <<  "Elevation " << "," <<  "Roll" << "," <<  "trakSTAR Time (ms since epoch)" << "," <<  "Quality" << endl;    
    DOUBLE_POSITION_ANGLES_TIME_Q_RECORD record[8*4];
    DOUBLE_POSITION_ANGLES_TIME_Q_RECORD *pRecord = record;

    printf("Starting Data Collection \nPress Q to stop data collection \n");

    while(1) {
		errorCode = GetSynchronousRecord(ALL_SENSORS, pRecord, sizeof(record)[0] * ATC3DG.m_config.numberSensors);
		if(errorCode!=BIRD_ERROR_SUCCESS) errorHandler(errorCode);

	


		// scan the sensors and request a record if the sensor is physically attached
		for(sensorID=0; sensorID < ATC3DG.m_config.numberSensors; sensorID++)
		{
			// get the status of the last data record and only report the data if everything is okay
			unsigned int status = GetSensorStatus( sensorID);

			if (status == VALID_STATUS)
			{
                string time_str = to_string(record[sensorID].time); //to format trakSTAR time into string so it goes in properly
                myFile << sensorID << "," << status << "," <<  record[sensorID].x  << "," <<  record[sensorID].y << "," <<  record[sensorID].z << "," <<  record[sensorID].a << "," <<  record[sensorID].e << "," <<  record[sensorID].r << "," <<  time_str << "," <<  record[sensorID].quality << endl;
				// save output to file. All data is stored in the record matrix
			}  
		}

        if (GetKeyState('Q') & 0x8000) { //checking high bit is 1 (1 << 15). If q is pressed, go to end routine
            printf("Stopped Data Collection. Shutting Down\n\n\n Thank you for using DATA COLLECTION APP. This window will close in 10 seconds");
            myFile.close();

            //for communication with python script
            ofstream updateFile("updateFile.txt");
            updateFile << "end";
            updateFile.close();




            /*
            System shut down (set transmitter to -1) -- SHRISHA
            */
            USHORT id = -1;
            errorCode = SetSystemParameter(SELECT_TRANSMITTER, &id, sizeof(id));
            if(errorCode!=BIRD_ERROR_SUCCESS) errorHandler(errorCode);

            sleep(10);

            return 0; // End function
        }

	}

}