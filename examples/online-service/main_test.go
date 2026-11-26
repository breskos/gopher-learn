package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func run(t *testing.T, name string, rows int) summary {
	t.Helper()
	spec := specs[name]
	r, err := open("", name, rows)
	if err != nil {
		t.Fatal(err)
	}
	events, err := readEvents(r, spec)
	if err != nil {
		t.Fatal(err)
	}
	split := len(events) / 5
	s, frozen, err := bootstrap(events[:split], spec)
	if err != nil {
		t.Fatal(err)
	}
	var out strings.Builder
	sum, err := replay(&out, s, frozen, events[split:], spec)
	if err != nil {
		t.Fatal(err)
	}
	t.Log("\n" + out.String())
	if sum.feedbackErrors > 0 {
		t.Fatalf("%d feedback errors", sum.feedbackErrors)
	}
	return sum
}

func TestOffers(t *testing.T) {
	sum := run(t, "offers", 40_000)
	if sum.live < 0.85 || sum.live < sum.frozen+0.2 || sum.stats.Drifts == 0 || sum.stats.Corrections == 0 {
		t.Fatalf("live %.3f, frozen %.3f, %+v", sum.live, sum.frozen, sum.stats)
	}
}

func TestDeliveries(t *testing.T) {
	sum := run(t, "deliveries", 40_000)
	if sum.live > 6 || sum.live > sum.frozen/2 || sum.stats.Rejected < 100 {
		t.Fatalf("live MAE %.2f, frozen %.2f, %+v", sum.live, sum.frozen, sum.stats)
	}
}

func TestHTTP(t *testing.T) {
	spec := specs["deliveries"]
	r, _ := open("", "deliveries", 3000)
	events, err := readEvents(r, spec)
	if err != nil {
		t.Fatal(err)
	}
	s, _, err := bootstrap(events, spec)
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(handler(s, spec))
	defer srv.Close()
	post := func(path, body string) (int, string) {
		resp, err := http.Post(srv.URL+path, "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = resp.Body.Close() }()
		b, _ := io.ReadAll(resp.Body)
		return resp.StatusCode, string(b)
	}
	for _, c := range []struct {
		path, body string
		code       int
		contains   string
	}{
		{"/predict", `{"id": "order-1", "features": [5, 3, 0.3, 12, 0]}`, 200, `"id":"order-1"`},
		{"/predict", `{"id": "order-1", "features": [5, 3, 0.3, 12, 0]}`, 409, "duplicate"},
		{"/predict", `{"features": [5, 3]}`, 400, "dimension"},
		{"/predict", `{"features": `, 400, "error"},
		{"/feedback", `{"id": "order-1", "value": 3600}`, 422, "implausible"},
		{"/feedback", `{"id": "order-1", "value": 31.5}`, 200, `"Version"`},
		{"/feedback", `{"id": "order-1", "value": 33}`, 200, `"Corrected":true`},
		{"/feedback", `{"id": "nope", "value": 33}`, 404, "unknown"},
		{"/feedback", `{"id": "order-1"}`, 400, "value missing"},
	} {
		code, body := post(c.path, c.body)
		if code != c.code || !strings.Contains(body, c.contains) {
			t.Errorf("%s %s: %d %s", c.path, c.body, code, body)
		}
	}
	resp, err := http.Get(srv.URL + "/stats")
	if err != nil {
		t.Fatal(err)
	}
	b, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if !strings.Contains(string(b), `"Corrections":1`) {
		t.Fatalf("stats %s", b)
	}
}
