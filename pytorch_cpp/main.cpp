#include <torch/torch.h>
#include <iostream>

int main() {
    torch::Tensor tensor = torch::rand({5, 10});
    std::cout << tensor << std::endl;
}
