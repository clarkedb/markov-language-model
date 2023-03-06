package random

import (
	"errors"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// ArgMaxMultinomial returns the index sampled by multinomial distribution with given probabilities.
func ArgMaxMultinomial(r *rand.Rand, pvals mat.Vector) (int, error) {
	v := 0.0
	p := r.Float64()
	for i := 0; i < pvals.Len(); i++ {
		v += pvals.AtVec(i)
		if v > p {
			return i, nil
		}
	}
	return -1, errors.New("invalid pvals")
}
