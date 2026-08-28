

const int N = 10;
const int counterMax = 10;
const float J = 1;
const int dimension = 2;
const int totalSpins = pow(N,dimension);
const double criticalTemperature = 2/(log(1+sqrt(2)));
const int rows = 1000;

#pragma pack(push, 1)
struct BMPHeader {
    char signature[2] = {'B', 'M'};
    uint32_t fileSize;      // Size of the BMP file
    uint16_t reserved1 = 0; // Reserved
    uint16_t reserved2 = 0; // Reserved
    uint32_t dataOffset;    // Offset to the start of image data
    uint32_t headerSize = 40; // Size of the header
    int32_t width;          // Width of the image
    int32_t height;         // Height of the image
    uint16_t planes = 1;
    uint16_t bitsPerPixel = 1; // 1-bit BMP
    uint32_t compression = 0;
    uint32_t dataSize = 0;
    int32_t horizontalResolution = 2835; // Pixels per meter (2835 ppm = 72 dpi)
    int32_t verticalResolution = 2835;   // Pixels per meter (2835 ppm = 72 dpi)
    uint32_t colors = 0;
    uint32_t importantColors = 0;
};
#pragma pack(pop)
void createBMP(const vector<vector<int>>& image, const std::string& filename) {
    BMPHeader header;
    int width = image[0].size();
    int height = image.size();

    header.width = width;
    header.height = height;
    int rowWidth = (width + 31) / 32 * 4; // Each row should be a multiple of 4 bytes

    header.fileSize = sizeof(BMPHeader) + rowWidth * height;

    std::ofstream bmpFile(filename, std::ios::out | std::ios::binary);

    // Write the BMP header
    bmpFile.write(reinterpret_cast<char*>(&header), sizeof(header));

    // Write the pixel data (bottom-up)
    for (int y = height - 1; y >= 0; --y) {
        for (int x = 0; x < width; ++x) {
            uint8_t pixel = (image[y][x] == 1) ? 0xFF : 0x00; // White or Black
            bmpFile.write(reinterpret_cast<char*>(&pixel), 1);
        }

        // Add padding to make the row width a multiple of 4 bytes
        for (int padding = rowWidth - width; padding > 0; --padding) {
            uint8_t paddingByte = 0x00;
            bmpFile.write(reinterpret_cast<char*>(&paddingByte), 1);
        }
    }

    bmpFile.close();
}
float* linspace(int arrayLength, float startValue, float endValue) {
    
     
    // Allocate memory for new array of length m
    float* interpArray = new float[arrayLength];
    
    // Iterate through the elements of new array giving equally m spaced values between start and end values, inclusive
    for (int n = 0; n < arrayLength; n++) {
        interpArray[n] = startValue + (endValue-startValue) * n / (arrayLength - 1);
    }
    
    return interpArray;
}
int getElement(vector<int> matrix, int address) {
    return matrix[address];
}
void setElement(vector<int> &matrix, int address, int val) {
    matrix[address] = val;
}

void initializeLattice(vector<int> &lattice) {
    random_device rd;
    mt19937 mt(rd());
    uniform_int_distribution<int> coin(0, 1);

    for (int i = 0; i < totalSpins; i++) {
        int spin = (coin(mt) == 0) ? -1 : 1;
        setElement(lattice, i, spin);
    }
}

vector<int> getNeighbours(int site) {
    vector<int> neighbours ={};

    int posSite, negSite;

    for (int d = 1;d <= dimension; d++) {
        
        posSite = int(pow(N,d)) * (site / int(pow(N,d)) ) + (site + int(pow(N,d-1)) + int(pow(N,dimension)) ) % ( int(pow(N,d)) );
        negSite = int(pow(N,d)) * (site / int(pow(N,d)) ) + (site - int(pow(N,d-1)) + int(pow(N,dimension)) ) % ( int(pow(N,d)) );      
        
        neighbours.push_back(posSite);
        neighbours.push_back(negSite);
        
    }
    return neighbours;
}
void showLattice(vector<int> &lattice, bool showValues = false ) {
    
    for (int site = 0; site < totalSpins; site++) {
        
        for (int dim = dimension; dim > 0; dim--){
            if (site %(int(pow(N,dim))) == 0) {
                cout << "\n";
            }            
        } 

        if (showValues){
            
            if (lattice[site] < 0){
                cout << lattice[site] << ",";
            }
            else{
                cout << " " << lattice[site] << ",";
            }
        }
        else{
            if (lattice[site] == 1) {
                cout << "+ ";
            } 
            else {
                cout << "- ";
            }
        }
    }
    
    


    int plusCount = 0, minusCount = 0;
    int agrees = 0, disagrees = 0;

    double magnetisation = 0;
    for (int site = 0; site < totalSpins; site++) {
        
        magnetisation = magnetisation + getElement(lattice,site);
        vector<int> siteNeighbours = getNeighbours(site);

        for (int dir = 0; dir < siteNeighbours.size(); dir++) {
            if (lattice[site]*lattice[siteNeighbours[dir]] ==1) {
                agrees++;
            } 
            else{
                disagrees++;
            }
        } 
    }

    
   
    

    std::cout << endl;

    std::cout << "Plus count: " << plusCount << " " <<"Minus count: " << minusCount <<"\n";
    std::cout << "Agrees: " << agrees << " " <<"Disagrees: " << disagrees <<"\n";
    std::cout << "Magnetisation: " << magnetisation << " " <<"Mean Magnetisation: " << magnetisation / (totalSpins) <<"\n";
    
}
void flipSpins(vector<int> &lattice, int site) {
        lattice[site] = -1*lattice[site];
}
void flipSpins(vector<int> &lattice, vector<int> sites) {
    for (int i = 0; i < sites.size(); i++) {
        lattice[sites[i]] = -1*lattice[sites[i]];  // flip the spin
    }
}
vector<int> buildCluster(vector<int> &lattice, int startSite, float temperature ) {

    int startState = getElement(lattice,startSite);

    vector<int> cluster = {startSite};
    vector<int> stackOld = {startSite};



    while (not stackOld.empty()) {
            vector<int> stackNew = {};
            
            // For all members of stack Old
            for (int i=0;i<stackOld.size();i++) {
                
                // Get the neighbours
                vector<int> neighbs = getNeighbours(stackOld[i]);
                
                //For each neighbour
                for (int j=0;j<neighbs.size();j++) {
                    
                    //if it isn't in the cluster
                    if ( find(cluster.begin(), cluster.end(), neighbs[j]) != cluster.end() ) {
                    }
                    else {
                        // If same state as start
                        if (getElement(lattice, neighbs[j]) == startState) {
                            
                            if  ((double) rand() / (RAND_MAX) <        (1-exp(-2*J/temperature))){
                                stackNew.push_back(neighbs[j]);
                                cluster.push_back(neighbs[j]);
                            }
                        }
                    }
                }
                stackOld = stackNew;
            }

        }
    return cluster;
}
vector<float> getTemperatures(float lowTempLowCutoff,float lowTempHighCutoff,float highTempLowCutoff,float highTempHighCutoff){

    int halfRows;
    if(rows % 2 == 0){
        halfRows = rows/2;
    } else{
        halfRows = rows/2+1;
    }

    vector<float> temperatures(rows); 
    float* lowTemps = new float[halfRows];
    float* highTemps = new float[rows - halfRows];

    lowTemps = linspace(halfRows, lowTempLowCutoff, lowTempHighCutoff);
    highTemps = linspace(rows - halfRows, highTempLowCutoff, highTempHighCutoff);


    for (int n = 0; n < halfRows; n++){
        temperatures[n] = lowTemps[n];
    }
    for (int m = 0; m < rows-halfRows; m++){
        temperatures[m+halfRows] = highTemps[m];
    }

    return temperatures;
    delete lowTemps;
    delete highTemps;

}















int main()
{
    float lowTempLowCutoff = 1;
    float lowTempHighCutoff = 2;
    float highTempLowCutoff = 3;
    float highTempHighCutoff = 5;

    vector<float> temperatures = getTemperatures(lowTempLowCutoff, lowTempHighCutoff, highTempLowCutoff, highTempHighCutoff);





    vector<int> lattice(totalSpins,0);
    initializeLattice(lattice);
    int labels[rows];

    // Set up output vector:

    vector<vector<int>> output(rows,vector<int>(totalSpins,0));


    for (int t = 0; t < rows; t++){

        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<int> distrib(0, totalSpins-1);    
        

        float temperature = temperatures[t];

        if (temperature > criticalTemperature){
            labels[t] = 1;
        } else{
            labels[t] = 0;
        }

        initializeLattice(lattice);

        int startSite = distrib(gen);
        int startState;

        
        for (int counter = 0; counter < counterMax; counter++) {
            
            vector<int> cluster = buildCluster(lattice, startSite, temperature);
            
            flipSpins(lattice, cluster);
        }


        /*
        for (int site = 0; site < totalSpins; site++){
            output[t][site] = int((lattice[site]+1)/2);
            cout << output[t][site] << ", ";
        }
        cout << endl;
        */

       

    }
    cout << "Done!";

    string outAddress = "output.bmp";

    createBMP(output, "output.bmp" );

    std::cout << "BMP file created successfully." << std::endl;


    return 0;


}