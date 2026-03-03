#include "matrix.h"
#include "matrix_math.h"
#include "autograd.h"
#include <assert.h>
#include <immintrin.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#define ROWS 6
#define COLS 6
#define INT 101
int main(){
	srand(0);
	matrix* m1 = matrix_random_normal(ROWS, COLS, 0, 1, 1);
	matrix* m2 = matrix_random_normal(1, COLS, 0, 1, 1);
	matrix* m3 = matrix_add_rowwise(m1, m2);
	matrix_print(m3);



} 





