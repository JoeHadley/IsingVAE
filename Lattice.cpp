

#include <iostream>
#include <stdexcept>
using namespace std;




class lattice {
public:
    const int dim;      // Number of dimensions
    int* latticeDimensions; // Dynamically allocated array for dimensions
    int Ntot;
    double* lat; // The working variables of the lattice

    // Constructor
    lattice(int dim, const int* dimensions)
        : dim(dim) {
        if (dim <= 0) {
            throw std::invalid_argument("Number of dimensions must be greater than 0.");
        }

        // Allocate memory for the dimensions array
        latticeDimensions = new int[dim];

        // Copy the input dimensions into the class array
        for (int i = 0; i < dim; ++i) {
            latticeDimensions[i] = dimensions[i];
        }

        Ntot = 1;
        for (int i = 0; i < dim; ++i) {
            Ntot = Ntot*latticeDimensions[i];
        }
        std::cout << Ntot;
        
        
        // Allocate memory for the lattice array
        lat = new double[Ntot];
        for (int i = 0; i < Ntot; ++i) {
            lat[i] = 0.0;
        }



    }

    // Destructor to free allocated memory
    ~lattice() {
        delete[] latticeDimensions;
        delete[] lat;
    }

    // Print dimensions for debugging
    void printDimensions() const {
        std::cout << "Lattice dimensions (" << dim << "-D): ";
        for (int i = 0; i < dim; ++i) {
            std::cout << latticeDimensions[i] << " ";
        }
        std::cout << std::endl;
    }

    // Copy constructor
    lattice(const lattice& other)
        : dim(other.dim), Ntot(other.Ntot) {
        latticeDimensions = new int[dim];
        for (int i = 0; i < dim; ++i) {
            latticeDimensions[i] = other.latticeDimensions[i];
        }

        lat = new double[Ntot];
        for (int i = 0; i < Ntot; ++i) {
            lat[i] = other.lat[i];
        }
    }

    // Copy assignment operator
    lattice& operator=(const lattice& other) {
        if (this == &other) {
            return *this;
        }

        delete[] latticeDimensions;
        delete[] lat;

        latticeDimensions = new int[other.dim];
        for (int i = 0; i < other.dim; ++i) {
            latticeDimensions[i] = other.latticeDimensions[i];
        }

        Ntot = other.Ntot;

        lat = new double[Ntot];
        for (int i = 0; i < Ntot; ++i) {
            lat[i] = other.lat[i];
        }

        return *this;
    }

    // Move constructor
    lattice(lattice&& other) noexcept
        : dim(other.dim), Ntot(other.Ntot), 
        latticeDimensions(other.latticeDimensions), lat(other.lat) {
        other.latticeDimensions = nullptr;
        other.lat = nullptr;
    }
    

    // Move assignment operator
    lattice& operator=(lattice&& other) noexcept {
        if (this == &other) {
            return *this;
        }

        // Free existing memory
        delete[] latticeDimensions;
        delete[] lat;

        // Steal resources
        latticeDimensions = other.latticeDimensions;
        lat = other.lat;
        Ntot = other.Ntot;

        other.latticeDimensions = nullptr;
        other.lat = nullptr;

        return *this;
    }
};

int main() {
    // Example usage
    int dim = 2;
    int dimensions[] = {4, 5};

    try {
        lattice myLattice(dim, dimensions);
        myLattice.printDimensions();

        // Test copy constructor
        //lattice copiedLattice = myLattice;
        //copiedLattice.printDimensions();

        // Test move constructor
        //lattice movedLattice = std::move(myLattice);
        //movedLattice.printDimensions();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}




