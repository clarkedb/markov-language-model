package random

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestArgMaxMultinomial(t *testing.T) {
	testR := rand.New(rand.NewSource(1))

	t.Run("uniform distribution", func(t *testing.T) {
		pvals := mat.NewVecDense(5, []float64{0.2, 0.2, 0.2, 0.2, 0.2})
		i, err := ArgMaxMultinomial(testR, pvals)
		assert.NoError(t, err)
		assert.GreaterOrEqual(t, i, 0)
		assert.Less(t, i, pvals.Len())
	})

	t.Run("non-uniform distribution", func(t *testing.T) {
		pvals := mat.NewVecDense(3, []float64{0.1, 0.6, 0.3})
		i, err := ArgMaxMultinomial(testR, pvals)
		assert.NoError(t, err)
		assert.GreaterOrEqual(t, i, 0)
		assert.Less(t, i, pvals.Len())
	})

	t.Run("sum of probabilities is less than 1", func(t *testing.T) {
		pvals := mat.NewVecDense(2, []float64{0.1, 0.2})
		i, err := ArgMaxMultinomial(testR, pvals)
		assert.Error(t, err)
		assert.Equal(t, -1, i)
	})
}
