#include <iostream>
#include <cmath>

using namespace std;

class actFunc 
{
    public:
    static  double applySigmoid(double in)
    {
        return 1.0/(1.0 + exp(-in));
    }
};