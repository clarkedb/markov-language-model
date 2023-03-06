package markov

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

var testR = rand.New(rand.NewSource(1))

func TestNewMarkovChain(t *testing.T) {
	type args struct {
		name      string
		matrix    *mat.Dense
		states    []string
		shouldErr bool
	}

	tests := []args{
		{
			name:      "square stochastic matrix with correct number of states initializes",
			matrix:    mat.NewDense(2, 2, []float64{0.5, 0.8, 0.5, 0.2}),
			states:    []string{"A", "B"},
			shouldErr: false,
		},
		{
			name:      "square stochastic matrix with incorrect number of states errors",
			matrix:    mat.NewDense(2, 2, []float64{0.5, 0.8, 0.5, 0.2}),
			states:    []string{"A", "B", "C"},
			shouldErr: false,
		},
		{
			name:      "non-square matrix errors",
			matrix:    mat.NewDense(2, 3, []float64{0.5, 0.8, 0.5, 0.2, 0.0, 0.0}),
			states:    []string{"A", "B"},
			shouldErr: true,
		},
		{
			name:      "square non-stochastic matrix with correct number of states errors",
			matrix:    mat.NewDense(2, 2, []float64{0.5, 0.7, 0.4, 0.1}),
			states:    []string{"A", "B"},
			shouldErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mc, err := NewMarkovChain(tt.matrix, tt.states, testR)

			if tt.shouldErr {
				assert.Error(t, err)
				assert.Nil(t, mc)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, mc)
			}
		})
	}
}

func TestTransition(t *testing.T) {
	matrix := mat.NewDense(2, 2, []float64{0.5, 0.8, 0.5, 0.2})
	states := []string{"A", "B"}
	mc, err := NewMarkovChain(matrix, states, testR)
	assert.NoError(t, err)
	assert.NotNil(t, mc)

	t.Run("it transitions", func(t *testing.T) {
		next, err := mc.Transition(states[0])
		assert.NoError(t, err)
		assert.Contains(t, states, next)

		next, err = mc.Transition(states[1])
		assert.NoError(t, err)
		assert.Contains(t, states, next)
	})
}

func TestWalk(t *testing.T) {
	matrix := mat.NewDense(2, 2, []float64{0.5, 0.8, 0.5, 0.2})
	states := []string{"A", "B"}
	mc, err := NewMarkovChain(matrix, states, testR)
	assert.NoError(t, err)
	assert.NotNil(t, mc)

	t.Run("it walks", func(t *testing.T) {
		n := 10
		walk, err := mc.Walk(states[0], n)
		assert.NoError(t, err)
		assert.Len(t, walk, n)
		assert.Equal(t, states[0], walk[0])
		assert.Condition(t, func() (success bool) {
			for _, s := range walk {
				assert.Contains(t, states, s)
			}
			return true
		})

		walk, err = mc.Walk(states[1], 10)
		assert.NoError(t, err)
		assert.Len(t, walk, n)
		assert.Equal(t, states[1], walk[0])
		assert.Condition(t, func() (success bool) {
			for _, s := range walk {
				assert.Contains(t, states, s)
			}
			return true
		})
	})
}

func TestPath(t *testing.T) {
	matrix := mat.NewDense(2, 2, []float64{0.5, 0.8, 0.5, 0.2})
	states := []string{"A", "B"}
	mc, err := NewMarkovChain(matrix, states, testR)
	assert.NoError(t, err)
	assert.NotNil(t, mc)

	t.Run("it computes a path", func(t *testing.T) {
		path, err := mc.Path(states[0], states[1])
		assert.NoError(t, err)
		assert.NotEmpty(t, path)
		assert.Equal(t, states[0], path[0])
		assert.Equal(t, states[1], path[len(path)-1])

		path, err = mc.Path(states[1], states[0])
		assert.NoError(t, err)
		assert.NotEmpty(t, path)
		assert.Equal(t, states[1], path[0])
		assert.Equal(t, states[0], path[len(path)-1])

		path, err = mc.Path(states[0], states[0])
		assert.NoError(t, err)
		assert.NotEmpty(t, path)
		assert.Equal(t, states[0], path[0])
		assert.Equal(t, states[0], path[len(path)-1])
	})

	t.Run("it errors if either state does not exist", func(t *testing.T) {
		path, err := mc.Path(states[0], "C")
		assert.Error(t, err)
		assert.Empty(t, path)

		path, err = mc.Path("D", states[1])
		assert.Error(t, err)
		assert.Empty(t, path)
	})
}

func TestSteadyState(t *testing.T) {
	type args struct {
		name           string
		matrix         *mat.Dense
		steadyState    []float64
		shouldConverge bool
	}

	tests := []args{
		{
			name:           "convergent matrix",
			matrix:         mat.NewDense(2, 2, []float64{0.5, 0.8, 0.5, 0.2}),
			steadyState:    []float64{0.6153846144112385, 0.3846153855887618},
			shouldConverge: true,
		},
		{
			name:           "periodic matrix",
			matrix:         mat.NewDense(2, 2, []float64{0.0, 1.0, 1.0, 0.0}),
			shouldConverge: false,
		},
		{
			name:           "identity matrix",
			matrix:         mat.NewDense(2, 2, []float64{0.8, 0.5, 0.2, 0.5}),
			steadyState:    []float64{0.71428571311, 0.2857142869},
			shouldConverge: true,
		},
	}
	states := []string{"A", "B"}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mc, err := NewMarkovChain(tt.matrix, states, nil)
			assert.NoError(t, err)
			assert.NotNil(t, mc)

			steady, ok := mc.SteadyState(1e-8, 1000)

			assert.Equal(t, tt.shouldConverge, ok)

			if tt.shouldConverge {
				assert.NotNil(t, steady)
				assert.InDeltaSlice(t, tt.steadyState, steady, 1e-7)
			}
		})
	}
}
