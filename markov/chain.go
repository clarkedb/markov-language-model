package markov

import (
	"fmt"
	"markovchains/linalg"
	"markovchains/random"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

type MarkovChain struct {
	TransitionMatrix *mat.Dense
	States           []string
	StatesToIndex    map[string]int
	r                *rand.Rand
}

func NewMarkovChain(transistionMatrix *mat.Dense, states []string, r *rand.Rand) (*MarkovChain, error) {
	m, n := transistionMatrix.Dims()

	// Check that A is square and column stochastic
	if m != n {
		return nil, fmt.Errorf("the transition matrix is not square")
	}

	columnSums := make([]float64, n)
	for j := 0; j < n; j++ {
		for i := 0; i < m; i++ {
			columnSums[j] += transistionMatrix.At(i, j)
		}
	}
	ones := make([]float64, n)
	for i := 0; i < n; i++ {
		ones[i] = 1.0
	}
	if !linalg.IsClose(columnSums, ones, 1e-8) {
		return nil, fmt.Errorf("the transistion matrix is not column stochastic")
	}

	// Initialize MarkovChain
	if r == nil {
		r = rand.New(rand.NewSource(time.Now().UnixNano()))
	}
	mc := &MarkovChain{
		TransitionMatrix: transistionMatrix,
		States:           states,
		StatesToIndex:    make(map[string]int),
		r:                r,
	}

	// Create state-index mapping
	for i, state := range states {
		mc.StatesToIndex[state] = i
	}

	return mc, nil
}

func (mc *MarkovChain) Transition(state string) (string, error) {
	// Check if the state value is a valid label
	j, ok := mc.StatesToIndex[state]
	if !ok {
		return "", fmt.Errorf("no such state %s found", state)
	}

	// Get next state
	probs := mc.TransitionMatrix.ColView(j)

	draw, err := random.ArgMaxMultinomial(mc.r, probs)
	if err != nil {
		return "", err
	}
	nextState := mc.States[draw]

	return nextState, nil
}

func (mc *MarkovChain) Walk(start string, n int) ([]string, error) {
	state := start
	states := []string{start}

	for i := 1; i < n; i++ {
		next, err := mc.Transition(state)
		if err != nil {
			return nil, err
		}
		state = next
		states = append(states, state)
	}

	return states, nil

}

func (mc *MarkovChain) Path(start, stop string) ([]string, error) {
	if _, ok := mc.StatesToIndex[start]; !ok {
		return []string{}, fmt.Errorf("no such start state %s found", start)
	}
	if _, ok := mc.StatesToIndex[stop]; !ok {
		return []string{}, fmt.Errorf("no such stop state %s found", stop)
	}

	state := start
	states := []string{start}

	for {
		next, err := mc.Transition(state)
		if err != nil {
			return nil, err
		}
		state = next
		states = append(states, state)

		if state == stop {
			break
		}
	}

	return states, nil
}

func (mc *MarkovChain) SteadyState(tol float64, maxiter int) ([]float64, bool) {
	n, _ := mc.TransitionMatrix.Dims()

	// Generate random state distribution vector
	xK1 := mat.NewVecDense(n, nil)
	for i := 0; i < n; i++ {
		xK1.SetVec(i, mc.r.Float64())
	}
	xK1 = linalg.NormalizeVec(xK1)
	xK := mat.NewVecDense(n, nil)

	k := 0

	for {
		// If we have exceeded maxiter, we assume non-convergence
		if k > maxiter {
			return nil, false
		}

		// Calculate next state distribution vector
		xK.MulVec(mc.TransitionMatrix, xK1)

		// If we are in tolerance level of convergence, break
		delta := mat.NewVecDense(xK.Len(), nil)
		delta.SubVec(xK, xK1)
		norm := delta.Norm(1)
		if norm < tol {
			break
		}

		// Else iterate
		xK1.CopyVec(xK)
		k++
	}

	// At this point we have the steady state
	return xK.RawVector().Data, true
}
