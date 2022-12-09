#include <iostream>

using namespace std;

void f1(int x) {
    x += 1;
    if (x == 1) {
        x -= 2;
        x /= 5;
    }
    if (x == 5) {
        x *= 4;
    }
    cout << x << endl;
}


void f2(int x) {
    x += 1;
    if (x == 1) {
        x -= 2;
        x /= 5;
    }
    if (x == 5) {
        x *= 4;
    }
    cout << x << endl;
}

// void f2(int x) {
//     x += 1;
//     if (x == 1) {
//         x -= 2;
//         if (x == 5) {
//             x /= 5;
//         }
//     }
//     x *= 4;
//     cout << x << endl;
// }

int main() {
    f1(1);
    f2(2);
}