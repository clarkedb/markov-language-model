package main

import (
	"markovchains/markov"
	"os"
	"strings"

	"gonum.org/v1/gonum/mat"
)

type SentenceGenerator struct {
	*markov.MarkovChain
}

const (
	StartToken = "$tart"
	StopToken  = "$top"
)

// TODO: refactor to take generic io.Reader?
func NewSentenceGenerator(filename string) *SentenceGenerator {
	s, err := os.ReadFile(filename)
	if err != nil {
		panic(err)
	}
	allWords := strings.Split(strings.ReplaceAll(string(s), "\n", " "), " ")
	sentences := strings.Split(string(s), "\n")

	// Find unique words (state labels)
	uniqueWords := make(map[string]bool)
	for _, word := range allWords {
		uniqueWords[word] = true
	}

	n := len(uniqueWords) + 2
	states := make([]string, n)
	states[0] = StartToken
	i := 1
	for word := range uniqueWords {
		states[i] = word
		i++
	}
	states[n-1] = StopToken

	wordsToIndexMap := make(map[string]int)
	transitionMatrix := make([]float64, n*n)

	// Create state-index mapping
	for i, state := range states {
		wordsToIndexMap[state] = i
	}

	// Populate the matrix
	for _, sentence := range sentences {
		words := append([]string{StartToken}, strings.Split(sentence, " ")...)
		words = append(words, StopToken)

		// Compare consecutive words
		for i := 1; i < len(words); i++ {
			x := wordsToIndexMap[words[i-1]]
			y := wordsToIndexMap[words[i]]
			transitionMatrix[(y*n)+x] += 1
		}
	}

	transitionMatrix[(n*n)-1] = 1 // $top -> $top

	// Normalize column sums
	for j := 0; j < n; j++ {
		colSum := 0.0
		for i := 0; i < n; i++ {
			colSum += transitionMatrix[(i*n)+j]
		}
		for i := 0; i < n; i++ {
			transitionMatrix[(i*n)+j] /= colSum
		}
	}

	tm := mat.NewDense(n, n, transitionMatrix)

	mc, err := markov.NewMarkovChain(tm, states, nil)
	if err != nil {
		panic(err)
	}
	sg := &SentenceGenerator{mc}

	return sg
}

func (sg *SentenceGenerator) Babble() (string, error) {
	// Get path
	words, err := sg.Path(StartToken, StopToken)
	if err != nil {
		return "", err
	}

	sentence := strings.Join(words[1:len(words)-1], " ")
	return sentence, nil
}
