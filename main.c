#include "matrix.h"
#include "matrix_math.h"
#include "autograd.h"
#include <assert.h>
#include <immintrin.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#define ROWS 10
#define COLS 10
#define INT 101
int main(){
	srand(0);
	matrix* m1 = matrix_random_normal(ROWS, COLS, 0, 1, 1);
	matrix* m2 = matrix_random_uniform(ROWS, COLS, -1, 1, 1);
	matrix* out = matrix_softmax(m2);
	matrix_backward(out);
	matrix_print(matrix_from_raw(m2->grad, m2->rows, m2->cols));
} 






