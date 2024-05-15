#include "Param_Progonka.h"
#include "Header.h"

void progonka() {
	for (int i = 1; i <= 4; i = i * 2) {
		for (int j = 512; j <= 4096; j = j * 2) {
			std::cout << std::endl << "Param progonka n=" << j << " p=" << i << std::endl;
			ParamProgonka(j, i);
		}
	}
	system("pause");
}
void cyclic_reductionStart() {
	for (int i = 1; i <= 4; i = i * 2) {
		for (int j = 9; j <= 12; j ++) {
			std::cout << std::endl << "Param progonka n=" << pow(2,j) << " p=" << i << std::endl;
			reduction_start(j, i);
		}
	}
	system("pause");
}

int main()
{	
	//progonka();
	cyclic_reductionStart();
}