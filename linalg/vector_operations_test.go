package linalg

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
)

func TestIsClose(t *testing.T) {
	type args struct {
		name  string
		a     []float64
		b     []float64
		tol   float64
		close bool
	}

	tests := []args{
		{
			name:  "",
			a:     []float64{1.0, 2.0, 3.0},
			b:     []float64{1.0, 2.0, 3.0},
			tol:   0.0,
			close: true,
		},
		{
			name:  "same values with non-zero tolerance",
			a:     []float64{1.0, 2.0, 3.0},
			b:     []float64{1.01, 1.99, 3.0},
			tol:   0.02,
			close: true,
		},
		{
			name:  "different lengths",
			a:     []float64{1.0, 2.0, 3.0},
			b:     []float64{1.0, 2.0},
			tol:   0.0,
			close: false,
		},
		{
			name:  "values differ more than tolerance",
			a:     []float64{1.0, 2.0, 3.0},
			b:     []float64{1.0, 2.2, 3.0},
			tol:   0.01,
			close: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			close := IsClose(tt.a, tt.b, tt.tol)
			assert.Equal(t, tt.close, close)
		})
	}
}

func TestNormalizeVec(t *testing.T) {
	type args struct {
		name string
		v    *mat.VecDense
		want *mat.VecDense
	}

	tests := []args{
		{
			name: "normalizing positive values",
			v:    mat.NewVecDense(3, []float64{1.0, 2.0, 3.0}),
			want: mat.NewVecDense(3, []float64{1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0}),
		},
		{
			name: "normalizing negative values",
			v:    mat.NewVecDense(3, []float64{-1.0, 2.0, -3.0}),
			want: mat.NewVecDense(3, []float64{-1.0 / -2.0, 2.0 / -2.0, -3.0 / -2.0}),
		},
		{
			name: "normalizing zero values",
			v:    mat.NewVecDense(3, []float64{0.0, 0.0, 0.0}),
			want: mat.NewVecDense(3, []float64{0.0, 0.0, 0.0}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NormalizeVec(tt.v)
			assert.InDeltaSlice(t, got.RawVector().Data, tt.want.RawVector().Data, 1e-6)
		})
	}
}
