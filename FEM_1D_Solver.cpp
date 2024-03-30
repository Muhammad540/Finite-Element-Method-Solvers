/* 
This is FEM 1D code to solve the following differential equation: 
                        - d/dx [a(x) du/dx] + b(x) du/dx + c(x) * u = f(x)
where a(x) = 1, b(x) = 0, c(x) = -pi^2, f(x) = -x

The code uses linear lagrange shape functions and Gauss quadrature with 2 points.
The code also allows the user to use a custom mesh with the option to use a geometric progression to calculate the step sizes.
The code also allows the user to apply Dirichlet and Neumann boundary conditions.
The code writes the solution to a file called solution.txt.
Author
    Name: Muhammad Ahmed
*/
#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <variant>
#include <string>
#include <fstream>
#include <iomanip>
using namespace std;


// Global variables
int nelem = 15; // Number of elements
int nen = 2;   // Number of nodes per element
int ngp = 2;   // Number of Gauss points per element
double xmin = 0.0;  // Minimum x value
double xmax = 3.0;  // Maximum x value
/* General form of DE: - d/dx [a(x) du/dx] + b(x) du/dx + c(x) * u = f(x) */
double a = 1.0; // Coefficient of second derivative
double b = 0.0; // Coefficient of first derivative
const double PI = 3.14159265358979323846;
double c = -PI*PI; // Coefficient of u
double f(double x) { return -x; } // Source term


//Setup Gauss quadrature
void setupGaussQuad(int ngp, vector <vector <double>> &gausspoints){
    if (ngp == 1){
        // point
        gausspoints[0][0] = 0.0;
        // weight
        gausspoints[0][1] = 2.0;
    }
    if (ngp == 2){
        // point
        gausspoints[0][0] =  -sqrt(1.0/3.0);
        gausspoints[1][0] =   sqrt(1.0/3.0);
        // weight
        gausspoints[0][1] = 1.0;
        gausspoints[1][1] = 1.0;
    }
    if (ngp == 3){
        // point
        gausspoints[0][0] = -sqrt(3.0/5.0);
        gausspoints[1][0] = 0.0;
        gausspoints[2][0] = sqrt(3.0/5.0);
        // weight
        gausspoints[0][1] = 5.0/9.0;
        gausspoints[1][1] = 8.0/9.0;
        gausspoints[2][1] = 5.0/9.0;
    }
}

// Calculate shape functions and their derivatives
void calShape(int nen, int ngp, vector <vector <double>> &gausspoints, Eigen::MatrixXd &S, Eigen::MatrixXd &dS){
    if (nen == 2 ) {  // Linear lagrange Shape Functions
        for ( int k = 0; k<ngp; k++){
            double ksi = gausspoints[k][0];
            S(0,k) = 0.5*(1-ksi);
            S(1,k) = 0.5*(1+ksi);
            dS(0,k) = -0.5;
            dS(1,k) = 0.5;
        }
    }
}

// calculate Elemental System
void calcElemSys( int e, int nen, int ngp,vector <double> &x, vector <vector <double>> &gausspoints, Eigen::MatrixXd &Ke , Eigen::VectorXd &Fe , Eigen::MatrixXi &loc2glob , Eigen::MatrixXd &S , Eigen::MatrixXd &dS){
    // Gauss Quadrature Loop
    for ( int k=0 ; k < ngp ; k++){
        double ksi = gausspoints[k][0];
        // Calculate the x value inside the element that corresponds to the gauss point
        double xfirst; // x value of the first node of the element
        double xlast; // x value of the last node of the element
        xfirst = x[loc2glob(e,0)];
        xlast = x[loc2glob(e,nen-1)];
        double X = 0.5*(xfirst+xlast) + 0.5*(xlast-xfirst)*ksi;

        // Calculate the Jacobian
        double J = 0.5*(xlast-xfirst);

        // Calculate the stiffness matrix 
        for (int i=0 ; i<nen ; i++){
            for (int j=0 ; j<nen ; j++){    
                Ke(i,j) = Ke(i,j) + ((a*dS(i,k)/J * dS(j,k)/J + b*S(i,k) * dS(j,k)/J + c*S(i,k) * S(j,k)) * J * gausspoints[k][1]);
            }
        }
        // Calculate the force vector
        for (int i=0; i<nen; i++){
            Fe(i) = Fe(i) + S(i,k) * f(X) * J * gausspoints[k][1];
            }
        }
}

// Assemble the global stiffness matrix and the global force vector
void assemble(int e, int nen, Eigen::MatrixXd &Ke, Eigen::VectorXd &Fe, Eigen::MatrixXi &loc2glob, Eigen::MatrixXd &K, Eigen::VectorXd &F){
    for (int i =0; i<nen ; i++){
        // I is the global node corresponding to the local node i
        int I = loc2glob(e,i);

        for ( int j = 0; j<nen ; j++){
            // J is the global node corresponding to the local node j
            int J = loc2glob(e,j);
            // Assemble the global stiffness matrix
            K(I,J) = K(I,J) + Ke(i,j);
        }
    }
    for (int i = 0; i<nen ; i++){
        int I = loc2glob(e,i);
        // Assemble the global force vector
        F(I) = F(I) + Fe(i);
    }
}

// Apply boundary conditions
void applyBC( int nnode, Eigen::MatrixXd &K, Eigen::VectorXd &F, std::vector<std::vector<std::variant<std::string, double>>> &vec){
    for (int i = 0 ; i < 2 ; i++){
        int node;
        if (i == 0){
            node = 0;
        }
        else{
            node = nnode-1;  
        }
        // First row in bc represents the value of the Dirichlet boundary condition
        // Second row in bc represents the value of the Neumann boundary condition
        // Apply Dirichlet boundary condition at node
        if (std::holds_alternative<std::string>(vec[i][0])) {
            std::string bc = std::get<std::string>(vec[i][0]);
            if (bc == "EBC"){
                if (std::holds_alternative<double>(vec[i][1])) {
                    double value = std::get<double>(vec[i][1]);
                    F(node)  = value;
                    for (int j = 0; j<nnode ; j++){
                        K(node,j) = 0.0;
                    }
                    K(node,node) = 1.0;
                }
            }
            if (bc == "NBC"){
                if (std::holds_alternative<double>(vec[i][1])) {
                    double value = std::get<double>(vec[i][1]);
                    F(node) = F(node) + value;
                }
            }
        }

    }
}

void solve(Eigen::MatrixXd &K, Eigen::VectorXd &F, Eigen::VectorXd &u){
    // Solve the system of equations
    u = K.colPivHouseholderQr().solve(F);
}

// Custom sized mesh
void custom_size_mesh(int nnode, vector <double> &custom_x){
    double length = xmax - xmin;
    double first_last_custom_x {0.25};
    double center_custom_x {2*first_last_custom_x};

    int center_elem = nnode/2 - 1;
    custom_x[0] = 0;
    custom_x[nnode-1] = 3;
    double r {0.859647}; // Manually calculate the r = geometric ratio
    double a {first_last_custom_x};
    for (int i = 1; i <= center_elem; i++){
        custom_x[i] = custom_x[i-1] + a*pow(r,i-1);
    }
    custom_x[center_elem+1] = custom_x[center_elem] + center_custom_x;
    for (int i = nnode-2; i > center_elem+1; i--){
        custom_x[i] = custom_x[i+1] - a*pow(r, nnode-2-i);
    }
}

int main() {
    // Determine the number of nodes
    int nnode = nelem * (nen - 1) + 1;
    // Compute the nodal coordinates
    double elemLength {};
    elemLength = static_cast<double>(xmax - xmin) / nelem;
    vector <double> x;
    vector <double> custom_x(nnode);
    custom_size_mesh(nnode, custom_x);
    // ask user if they want to use custom mesh yes or no
    string want_custom_mesh;
    cout << "Do you want to use custom mesh? (yes/no): ";
    cin >> want_custom_mesh;
    if (want_custom_mesh == "yes"){
        x = custom_x;
    }
    for (double i = xmin; i <= xmax; i += elemLength) {
        x.push_back(i);
    }
    // Setup local to global node mapping array
    // row : number of elements : number of nodes per element (nen)
    Eigen::MatrixXi loc2glob(nelem, nen);
    for (int i = 0; i < nelem; i++) {
        for (int j = 0; j < nen; j++) {
            loc2glob(i, j) = ((i+1)-1)*(nen-1) + j;
        }
    }
    // print local to global node mapping array
    cout << "Local to global node mapping array: " << endl;
    cout << loc2glob << endl;

    // 2D vector to store gauss quadrature points
    vector<vector<double>> gausspoints ;
    gausspoints.resize(ngp, vector<double>(2));
    setupGaussQuad(ngp, gausspoints);

    // Define the shape functions and their derivatives
    Eigen::MatrixXd S(nen, ngp);
    Eigen::MatrixXd dS(nen, ngp);
    calShape(nen, ngp, gausspoints, S, dS);

    // Print the shape functions and their derivatives
    cout << "Shape functions: " << endl;
    cout << S << endl;
    cout << "Derivatives of shape functions: " << endl;
    cout << dS << endl;

    // Calculate the global stiffness matrix and the global force vector
    Eigen::MatrixXd K(nnode, nnode);
    Eigen::VectorXd F(nnode);
    K.setZero();
    F.setZero();


    // calculate elemental system
    for (int e = 0; e < nelem; e++) {
        Eigen::MatrixXd Ke(nen, nen);
        Eigen::VectorXd Fe(nen);
        Ke.setZero();
        Fe.setZero();
        calcElemSys(e, nen, ngp, x, gausspoints, Ke , Fe , loc2glob , S , dS);
        // Assemble the global stiffness matrix and the global force vector
        assemble(e, nen, Ke, Fe, loc2glob, K, F);
    }
    cout << "Global stiffness matrix before BC: " << endl;
    cout << K << endl;
    cout << "Global force vector before BC: " << endl;
    cout << F << endl;
    // Print the global stiffness matrix and the global force vector
    // Declare the boundary conditions vector
    using Cell = std::variant<std::string, double>;
    std::vector<std::vector<Cell>> vec = {
        { "EBC", 1.0 },
        { "NBC", 0.0 }
    };
    applyBC(nnode, K, F, vec);
    cout << "Global stiffness matrix after BC: " << endl;
    cout << K << endl;
    cout << "Global force vector after BC: " << endl;
    cout << F << endl;
    // solution vector
    Eigen::VectorXd u(nnode);
    // Solve the system of equations
    solve(K, F, u);
    cout << "Solution vector: " << endl;
    cout << u << endl;
    // write the solution to a file
    ofstream myfile;
    myfile.open("solution.txt");
    for ( const auto& value: u){
        myfile << value << endl;
    }
    myfile.close();
    
    return 0;
}
