#include <iostream>

class Data {
    private:
        static std::vector<std::vector<double>> data;
    public:
        static std::vector<std::vector<double>> getDataX();
        static std::vector<double> getDataY();
};