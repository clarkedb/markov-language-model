package main

import (
	"fmt"
)

func main() {
	sg := NewSentenceGenerator("./data/quotes.txt")

	for i := 0; i < 10; i++ {
		sen, err := sg.Babble()
		if err != nil {
			panic(err)
		}
		fmt.Println(sen)
	}
}
