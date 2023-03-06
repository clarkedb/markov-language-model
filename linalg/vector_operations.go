package linalg

import "gonum.org/v1/gonum/mat"

func IsClose(a, b []float64, tol float64) bool {
	if len(a) != len(b) {
		return false
	}

	for i := 0; i < len(a); i++ {
		if (a[i]-b[i]) > tol || (b[i]-a[i]) > tol {
			return false
		}
	}

	return true
}

func NormalizeVec(v *mat.VecDense) *mat.VecDense {
	sum := 0.0
	for i := 0; i < v.Len(); i++ {
		sum += v.AtVec(i)
	}
	if sum == 0.0 {
		return v
	}

	normalized := make([]float64, v.Len())
	for i := 0; i < v.Len(); i++ {
		normalized[i] = v.AtVec(i) / sum
	}

	nv := mat.NewVecDense(v.Len(), normalized)
	return nv
}
