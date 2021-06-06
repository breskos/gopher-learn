package encoders

import (
	"log"
	"regexp"
	"strings"
)

func normalizeString(value string) string {
	value = strings.ToLower(value)
	reg, err := regexp.Compile("[^a-zA-Z0-9]+")
	if err != nil {
		log.Fatal(err)
	}
	value = reg.ReplaceAllString(value, "")
	return value
}
